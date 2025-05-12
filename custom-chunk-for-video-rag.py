import json
import re
import os
import logging
import tempfile
import boto3
from typing import List, Dict, Any
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_transcript_for_embeddings(transcript_json: Dict[str, Any], video_file_path: str) -> List[Dict[str, Any]]:
    """
    Process AWS Transcribe output into chunks suitable for embeddings with timestamp metadata
    and video file location.

    Args:
        transcript_json: The AWS Transcribe output JSON
        video_file_path: Path or URL to the source video file

    Returns:
        List of dictionaries containing text chunks with metadata
    """

    logger.info(f"Processing transcript for video: {video_file_path}")

    # Extract video filename from path
    video_filename = os.path.basename(video_file_path)

    # Extract segments from the transcript
    segments = transcript_json.get("results", {}).get("audio_segments", [])
    if not segments:
        # Fall back to transcript items if segments aren't available
        logger.info("No audio_segments found, creating segments from items")
        full_transcript = transcript_json.get("results", {}).get("transcripts", [{}])[0].get("transcript", "")
        items = transcript_json.get("results", {}).get("items", [])

        # Create segments by topic or time gaps
        segments = create_segments_from_items(items, full_transcript)

    # Process each segment into an embedding-ready chunk
    embedding_chunks = []
    for i, segment in enumerate(segments):
        chunk = {
            "text": segment.get("transcript", ""),
            "metadata": {
                "video_file_path": video_file_path,
                "video_filename": video_filename,
                "start_time": float(segment.get("start_time", 0)),
                "end_time": float(segment.get("end_time", 0)),
                "segment_id": segment.get("id", "") or f"segment_{i}",
                "duration": float(segment.get("end_time", 0)) - float(segment.get("start_time", 0))
            }
        }

        # Clean the text
        chunk["text"] = clean_text(chunk["text"])

        # Only keep non-empty chunks of sufficient length
        if len(chunk["text"]) > 20:  # Adjust minimum length as needed
            embedding_chunks.append(chunk)

    logger.info(f"Created {len(embedding_chunks)} embedding chunks")
    return embedding_chunks


def clean_text(text: str) -> str:
    """Clean text for better embedding quality"""
    # Remove filler words, normalize whitespace, etc.
    text = re.sub(r'\b(um|uh|like|you know)\b', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def create_segments_from_items(items: List[Dict], transcript: str) -> List[Dict]:
    """Create logical segments from individual transcript items"""
    segments = []
    current_segment = {"transcript": "", "start_time": None, "end_time": None, "items": []}

    # Look for pauses or speaker changes to define segment boundaries
    for i, item in enumerate(items):
        if item.get("type") == "pronunciation":
            if current_segment["start_time"] is None:
                current_segment["start_time"] = item.get("start_time")

            current_segment["items"].append(item)
            current_segment["end_time"] = item.get("end_time")

            # Check for pause (time gap > 1.5 seconds) or maximum segment length
            if i < len(items) - 1 and items[i + 1].get("type") == "pronunciation":
                next_start = float(items[i + 1].get("start_time", 0))
                current_end = float(item.get("end_time", 0))
                segment_duration = current_end - float(current_segment["start_time"])

                # Split on significant pause or if segment is getting too long (>30 seconds)
                if next_start - current_end > 1.5 or segment_duration > 30:
                    # Finalize current segment
                    current_segment["transcript"] = build_segment_text(current_segment["items"])
                    segments.append(current_segment)
                    current_segment = {"transcript": "", "start_time": None, "end_time": None, "items": []}

    # Add final segment if not empty
    if current_segment["items"]:
        current_segment["transcript"] = build_segment_text(current_segment["items"])
        segments.append(current_segment)

    return segments


def build_segment_text(items: List[Dict]) -> str:
    """Build segment text from items"""
    text = ""
    for item in items:
        if item.get("type") == "pronunciation":
            text += item.get("alternatives", [{}])[0].get("content", "") + " "
        elif item.get("type") == "punctuation":
            text = text.strip() + item.get("alternatives", [{}])[0].get("content", "") + " "

    return text.strip()


def merge_chunks_into_segments(chunks: List[Dict], max_duration_seconds: int = 180) -> List[Dict]:
    """
    Sort chunks by start time and merge them into larger segments that don't exceed the specified duration

    Args:
        chunks: List of transcript chunks with metadata
        max_duration_seconds: Maximum duration for merged segments (default: 180 seconds = 3 minutes)

    Returns:
        List of merged chunks with updated metadata
    """
    logger.info(f"Merging {len(chunks)} chunks into segments of max {max_duration_seconds} seconds")

    # Sort chunks by start time
    sorted_chunks = sorted(chunks, key=lambda x: x["metadata"]["start_time"])

    merged_chunks = []
    current_segment = None

    for chunk in sorted_chunks:
        # If we don't have a current segment or adding this chunk would exceed max duration,
        # create a new segment
        if (current_segment is None or
                (chunk["metadata"]["end_time"] - current_segment["metadata"]["start_time"]) > max_duration_seconds):

            if current_segment is not None:
                merged_chunks.append(current_segment)

            # Start a new segment with this chunk
            current_segment = {
                "text": chunk["text"],
                "metadata": {
                    "video_file_path": chunk["metadata"]["video_file_path"],
                    "video_filename": chunk["metadata"]["video_filename"],
                    "start_time": chunk["metadata"]["start_time"],
                    "end_time": chunk["metadata"]["end_time"],
                    "segment_id": f"merged_{len(merged_chunks)}",
                    "duration": chunk["metadata"]["duration"]
                }
            }
        else:
            # Merge this chunk into the current segment
            current_segment["text"] += " " + chunk["text"]
            current_segment["metadata"]["end_time"] = chunk["metadata"]["end_time"]
            current_segment["metadata"]["duration"] = (
                    current_segment["metadata"]["end_time"] - current_segment["metadata"]["start_time"]
            )

    # Add the last segment if there is one
    if current_segment is not None:
        merged_chunks.append(current_segment)

    logger.info(f"Created {len(merged_chunks)} merged segments")
    return merged_chunks


def save_chunks_to_text_files(chunks: List[Dict], output_dir: str, include_metadata: bool = True) -> List[str]:
    """
    Save each chunk to a separate text file without transformation

    Args:
        chunks: List of transcript chunks with metadata
        output_dir: Directory where text files will be saved
        include_metadata: Whether to include metadata as a header in the text files

    Returns:
        List of paths to the created text files
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    file_paths = []

    for i, chunk in enumerate(chunks):
        # Create a filename based on video name and time range
        video_name = os.path.splitext(chunk["metadata"]["video_filename"])[0]
        start_time = chunk["metadata"]["start_time"]
        end_time = chunk["metadata"]["end_time"]

        # Format times as MM:SS
        start_time_fmt = f"{int(start_time // 60):02d}_{int(start_time % 60):02d}"
        end_time_fmt = f"{int(end_time // 60):02d}_{int(end_time % 60):02d}"

        # Create filename
        filename = f"{video_name}_{start_time_fmt}_to_{end_time_fmt}.txt"
        file_path = os.path.join(output_dir, filename)

        # Just write the text directly to file without any modifications
        with open(file_path, 'w', encoding='utf-8') as f:
            # Add minimal metadata as a comment if requested
            if include_metadata:
                f.write(f"# Metadata:\n")
                for key, value in chunk["metadata"].items():
                    f.write(f"# {key}: {value}\n")
                f.write("#\n# Transcript:\n")

            # Write the raw text content as-is
            f.write(chunk["text"])

        file_paths.append(file_path)

    logger.info(f"Saved {len(file_paths)} text files to {output_dir}")
    return file_paths


def save_chunks_to_s3(chunks: List[Dict], s3_bucket: str, s3_prefix: str, include_metadata: bool = True) -> List[str]:
    """
    Save each chunk to a separate text file in S3 without transformation

    Args:
        chunks: List of transcript chunks with metadata
        s3_bucket: S3 bucket name
        s3_prefix: S3 key prefix (folder path)
        include_metadata: Whether to include metadata as a header in the text files

    Returns:
        List of S3 URIs for the uploaded files
    """
    s3_client = boto3.client('s3')
    s3_uris = []

    # Make sure the prefix ends with a slash if it's not empty
    if s3_prefix and not s3_prefix.endswith('/'):
        s3_prefix += '/'

    for i, chunk in enumerate(chunks):
        # Create a filename based on video name and time range
        video_name = os.path.splitext(chunk["metadata"]["video_filename"])[0]
        start_time = chunk["metadata"]["start_time"]
        end_time = chunk["metadata"]["end_time"]

        # Format times as MM:SS
        start_time_fmt = f"{int(start_time // 60):02d}_{int(start_time % 60):02d}"
        end_time_fmt = f"{int(end_time // 60):02d}_{int(end_time % 60):02d}"

        # Create filename
        filename = f"{video_name}_{start_time_fmt}_to_{end_time_fmt}.txt"
        s3_key = f"{s3_prefix}{filename}"

        # Create the file content
        content = ""
        if include_metadata:
            content += f"# Metadata:\n"
            for key, value in chunk["metadata"].items():
                content += f"# {key}: {value}\n"
            content += "#\n# Transcript:\n"

        # Add the transcript text
        content += chunk["text"]

        # Upload to S3
        try:
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=s3_key,
                Body=content,
                ContentType='text/plain'
            )
            s3_uri = f"s3://{s3_bucket}/{s3_key}"
            s3_uris.append(s3_uri)
            logger.debug(f"Uploaded chunk to {s3_uri}")
        except ClientError as e:
            logger.error(f"Error uploading chunk to S3: {e}")
            raise

    logger.info(f"Saved {len(s3_uris)} text files to S3 bucket {s3_bucket} with prefix {s3_prefix}")
    return s3_uris


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS"""
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


def format_duration(seconds: float) -> str:
    """Format duration in a human-readable format"""
    minutes, seconds = divmod(int(seconds), 60)

    if minutes > 0:
        return f"{minutes} min {seconds} sec"
    else:
        return f"{seconds} sec"


def download_from_s3(s3_bucket: str, s3_key: str, local_path: str = None):
    """
    Download a file from S3 to local storage

    Args:
        s3_bucket: S3 bucket name
        s3_key: S3 object key
        local_path: Local file path to save the downloaded file. If None, will use a temporary file.

    Returns:
        Path to the downloaded file
    """
    s3_client = boto3.client('s3')

    # If no local path is provided, create a temporary file
    if local_path is None:
        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(tempfile.gettempdir(), 'transcript_processor')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Use the original filename for the temp file
        filename = os.path.basename(s3_key)
        local_path = os.path.join(temp_dir, filename)

    # Create directory for the local file if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)

    try:
        logger.info(f"Downloading S3://{s3_bucket}/{s3_key} to {local_path}")
        s3_client.download_file(s3_bucket, s3_key, local_path)
        return local_path
    except ClientError as e:
        logger.error(f"Error downloading file from S3: {e}")
        raise


def process_transcript_from_s3(
        transcript_s3_bucket: str,
        transcript_s3_key: str,
        video_s3_bucket: str,
        video_s3_key: str,
        output_s3_bucket: str,
        output_s3_prefix: str,
        merge_segments: bool = False,
        max_duration_seconds: int = 180,
        save_json_to_s3: bool = True,
        save_text_files_to_s3: bool = True,
        include_metadata: bool = True
) -> List[Dict]:
    """
    Process a transcript file from S3 and store the results back to S3

    Args:
        transcript_s3_bucket: S3 bucket containing the transcript file
        transcript_s3_key: S3 key for the transcript file
        video_s3_bucket: S3 bucket containing the video file
        video_s3_key: S3 key for the video file
        output_s3_bucket: S3 bucket for output files
        output_s3_prefix: S3 prefix (folder) for output files
        merge_segments: Whether to merge chunks into larger segments
        max_duration_seconds: Maximum duration for merged segments if merge_segments is True
        save_json_to_s3: Whether to save the JSON output to S3
        save_text_files_to_s3: Whether to save individual text files to S3
        include_metadata: Whether to include metadata in text files

    Returns:
        List of processed chunks ready for embedding
    """
    try:
        # Create a temporary directory for local processing
        temp_dir = os.path.join(tempfile.gettempdir(), 'transcript_processor')
        os.makedirs(temp_dir, exist_ok=True)

        # Download the transcript file from S3
        transcript_local_path = download_from_s3(
            transcript_s3_bucket,
            transcript_s3_key,
            os.path.join(temp_dir, os.path.basename(transcript_s3_key))
        )

        # Construct the S3 path for the video
        video_s3_path = f"s3://{video_s3_bucket}/{video_s3_key}"

        # Load the transcript file
        logger.info(f"Loading transcript from {transcript_local_path}")
        with open(transcript_local_path, 'r') as f:
            transcript_json = json.load(f)

        # Process transcript into chunks
        chunks = process_transcript_for_embeddings(transcript_json, video_s3_path)

        # Optionally merge chunks into larger segments
        if merge_segments:
            chunks = merge_chunks_into_segments(chunks, max_duration_seconds)

        # Save the processed chunks to a JSON file in S3
        # if save_json_to_s3:
        #     json_output_key = f"{output_s3_prefix}/chunks.json"
        #     if not json_output_key.startswith('/'):
        #         json_output_key = f"{output_s3_prefix}/chunks.json"
        #
        #     try:
        #         s3_client = boto3.client('s3')
        #         s3_client.put_object(
        #             Bucket=output_s3_bucket,
        #             Key=json_output_key,
        #             Body=json.dumps(chunks, indent=2),
        #             ContentType='application/json'
        #         )
        #         logger.info(f"Saved {len(chunks)} processed chunks to S3://{output_s3_bucket}/{json_output_key}")
        #     except ClientError as e:
        #         logger.error(f"Error saving JSON to S3: {e}")
        #         raise

        # Optionally save individual text files for each chunk to S3
        if save_text_files_to_s3:
            text_files_prefix = f"{output_s3_prefix}"
            if text_files_prefix.endswith('/'):
                text_files_prefix = text_files_prefix[:-1]

            s3_uris = save_chunks_to_s3(
                chunks,
                output_s3_bucket,
                text_files_prefix,
                include_metadata
            )
            logger.info(f"Saved {len(s3_uris)} text files to S3")

        return chunks

    except Exception as e:
        logger.error(f"Error processing transcript from S3: {e}")
        raise


def process_transcript_file(transcript_file_path: str, video_file_path: str, merge_segments: bool = False,
                            max_duration_seconds: int = 180, save_text_files: bool = False,
                            output_dir: str = None) -> List[Dict]:
    """
    Process a transcript file and return chunks ready for embedding

    Args:
        transcript_file_path: Path to the transcript JSON file
        video_file_path: Path or URL to the source video file
        merge_segments: Whether to merge chunks into larger segments
        max_duration_seconds: Maximum duration for merged segments if merge_segments is True
        save_text_files: Whether to save chunks as individual text files
        output_dir: Directory where text files will be saved (if save_text_files is True)

    Returns:
        List of processed chunks ready for embedding
    """
    try:
        # Load the transcript file
        logger.info(f"Loading transcript from {transcript_file_path}")
        with open(transcript_file_path, 'r') as f:
            transcript_json = json.load(f)

        # Process transcript into chunks
        chunks = process_transcript_for_embeddings(transcript_json, video_file_path)

        # Optionally merge chunks into larger segments
        if merge_segments:
            chunks = merge_chunks_into_segments(chunks, max_duration_seconds)

        # Optionally save the processed chunks to a JSON file
        json_output_file = os.path.splitext(transcript_file_path)[0] + "_chunks.json"
        with open(json_output_file, 'w') as f:
            json.dump(chunks, f, indent=2)
        logger.info(f"Saved {len(chunks)} processed chunks to {json_output_file}")

        # Optionally save individual text files for each chunk
        if save_text_files:
            if output_dir is None:
                # Default output directory based on transcript filename
                output_dir = os.path.splitext(transcript_file_path)[0] + "_chunks"

            save_chunks_to_text_files(chunks, output_dir)

        return chunks

    except Exception as e:
        logger.error(f"Error processing transcript file: {str(e)}")
        raise


# Example usage with local files
def process_local_example():
    # Example file paths
    transcript_file = "./transcripts/adp-transcript.json"
    video_file = "./videos/adp-video.mp4"

    # Process the transcript with various options
    chunks = process_transcript_file(
        transcript_file_path=transcript_file,
        video_file_path=video_file,
        merge_segments=True,  # Merge into larger segments
        max_duration_seconds=180,  # 3 minutes max per segment
        save_text_files=True,  # Save as individual text files
        output_dir="adp_transcript_chunks"  # Directory for text files
    )

    print(f"Processed transcript into {len(chunks)} chunks")


# Example usage with S3 files
def process_s3_example():
    # Example S3 file paths
    transcript_s3_bucket = "jay-vid-rag-poc-2"
    transcript_s3_key = "adp-vid2.json"
    video_s3_bucket = "jay-vid-rag-poc-2"
    video_s3_key = "ADP_AI_AT_WORK.mp4"
    output_s3_bucket = "jay-kb-poc1"
    output_s3_prefix = ""

    # Process the transcript with various options
    chunks = process_transcript_from_s3(
        transcript_s3_bucket=transcript_s3_bucket,
        transcript_s3_key=transcript_s3_key,
        video_s3_bucket=video_s3_bucket,
        video_s3_key=video_s3_key,
        output_s3_bucket=output_s3_bucket,
        output_s3_prefix=output_s3_prefix,
        merge_segments=True,  # Merge into larger segments
        max_duration_seconds=180,  # 3 minutes max per segment
        save_json_to_s3=True,  # Save JSON output to S3
        save_text_files_to_s3=True,  # Save individual text files to S3
        include_metadata=True  # Include metadata in text files
    )

    print(f"Processed S3 transcript into {len(chunks)} chunks")


# AWS Lambda handler
def lambda_handler(event, context):
    """
    AWS Lambda handler function

    Args:
        event: Lambda event data, expected to contain:
            - transcript_s3_bucket: S3 bucket containing the transcript file
            - transcript_s3_key: S3 key for the transcript file
            - video_s3_bucket: S3 bucket containing the video file
            - video_s3_key: S3 key for the video file
            - output_s3_bucket: S3 bucket for output files
            - output_s3_prefix: S3 prefix (folder) for output files
            - merge_segments: Whether to merge chunks (optional, default: False)
            - max_duration_seconds: Maximum duration for merged segments (optional, default: 180)
        context: Lambda context

    Returns:
        Dictionary with status and number of chunks processed
    """
    try:
        # Extract parameters from the event
        transcript_s3_bucket = event.get('transcript_s3_bucket')
        transcript_s3_key = event.get('transcript_s3_key')
        video_s3_bucket = event.get('video_s3_bucket')
        video_s3_key = event.get('video_s3_key')
        output_s3_bucket = event.get('output_s3_bucket')
        output_s3_prefix = event.get('output_s3_prefix')
        merge_segments = event.get('merge_segments', False)
        max_duration_seconds = event.get('max_duration_seconds', 180)

        # Validate required parameters
        if not all([transcript_s3_bucket, transcript_s3_key, video_s3_bucket, video_s3_key,
                    output_s3_bucket, output_s3_prefix]):
            return {
                'statusCode': 400,
                'body': json.dumps('Missing required parameters')
            }

        # Process the transcript
        chunks = process_transcript_from_s3(
            transcript_s3_bucket=transcript_s3_bucket,
            transcript_s3_key=transcript_s3_key,
            video_s3_bucket=video_s3_bucket,
            video_s3_key=video_s3_key,
            output_s3_bucket=output_s3_bucket,
            output_s3_prefix=output_s3_prefix,
            merge_segments=merge_segments,
            max_duration_seconds=max_duration_seconds
        )

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Successfully processed transcript',
                'num_chunks': len(chunks),
                'output_s3_bucket': output_s3_bucket,
                'output_s3_prefix': output_s3_prefix
            })
        }

    except Exception as e:
        logger.error(f"Error in lambda handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error processing transcript: {str(e)}')
        }


# Main entry point
if __name__ == "__main__":
    # Uncomment the appropriate function for your use case
    # process_local_example()  # Process local files
    process_s3_example()  # Process S3 files