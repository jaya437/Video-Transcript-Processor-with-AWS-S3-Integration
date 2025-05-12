# Video Transcript Processor with AWS S3 Integration

This repository provides a Python-based solution for processing video transcript data from AWS Transcribe, generating text chunks for embedding, and saving them either locally or directly to an S3 bucket. The solution is designed to efficiently manage large transcript files by creating logically segmented text chunks with metadata, making them suitable for downstream tasks like natural language processing, search indexing, or machine learning.

## Key Features

* **Transcript Chunking:** Automatically segments the transcript into meaningful chunks based on speaker changes, pauses, or defined time intervals.
* **Text Cleaning:** Applies text normalization and cleaning for better embedding quality.
* **Segment Merging:** Optionally merges smaller chunks into larger segments without exceeding a specified duration.
* **Local and S3 Processing:** Supports processing of transcripts and videos from both local files and AWS S3 buckets.
* **S3 Upload:** Saves processed transcript chunks as text files in S3, with optional metadata headers.
* **Metadata Inclusion:** Adds metadata such as video file name, start and end times, and segment duration to each chunk.
* **AWS Lambda Compatible:** Includes a ready-to-use AWS Lambda handler for serverless transcript processing.

## How It Works

1. **Input:** The system accepts a transcript file (JSON) and an associated video file (MP4), either locally or from S3.
2. **Processing:** The transcript is segmented based on logical criteria, cleaned for embedding, and metadata is added.
3. **Output:** The processed chunks are saved as individual text files, either locally or in S3, with optional metadata.

## Setup Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/video-transcript-processor.git
   cd video-transcript-processor
   ```

2. Install required dependencies:

   ```bash
   pip install boto3
   ```

3. Configure AWS credentials using the AWS CLI:

   ```bash
   aws configure
   ```

4. Customize the S3 bucket names and paths in the `process_s3_example()` function or the AWS Lambda handler as needed.

## Usage

* **Local File Processing:**

  ```bash
  python video_transcript_processor.py
  ```

* **S3 File Processing:**
  Ensure that the S3 buckets and file paths are correctly configured in the `process_s3_example()` function.

* **AWS Lambda Deployment:** Package the code as a Lambda function and configure it with the appropriate S3 permissions.

## Example Use Cases

* Automatically process large video transcripts for embedding and search.
* Segment lecture transcripts for educational content indexing.
* Integrate with AWS Lambda for scalable, serverless transcript processing.

## License

This project is licensed under the MIT License.
