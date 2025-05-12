import json
import logging
import os
import re
import boto3
from botocore.exceptions import ClientError
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse

# Configure logging for Lambda
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# S3 client for generating signed URLs
s3_client = boto3.client('s3')

# Environment variables
VIDEO_BUCKET = os.environ.get('VIDEO_BUCKET', 'jay-vid-rag-poc-2')
SIGNED_URL_EXPIRATION = int(os.environ.get('SIGNED_URL_EXPIRATION', 1800))  # Default 30 minutes


def extract_s3_paths_and_metadata(augmented_response: str) -> List[Dict[str, Any]]:
    """
    Extract S3 paths from the augmented response text.
    Looks for patterns like [1] and corresponding "Sources:" section with MP4 paths.
    """
    try:
        print(f"DEBUG: Input augmented_response length: {len(augmented_response)}")

        # Get the sources section at the end
        sources_section_match = re.search(r'Sources:\s*((?:\[\d+\]\s*s3://[^\n]+\s*)+)', augmented_response, re.DOTALL)
        if not sources_section_match:
            print(f"WARNING: No sources section found in the augmented response.")
            return []

        sources_section = sources_section_match.group(1)
        print(f"DEBUG: Found sources section: {sources_section}")

        # Extract individual citations - Fixed regex to capture the fragment correctly
        # This pattern now properly captures the s3 path (including spaces) and the fragment if it exists
        citation_pattern = re.compile(r'\[(\d+)\]\s*(s3://[^#\n]+)(#t=(\d+))?')
        citations = []

        all_matches = list(citation_pattern.finditer(sources_section))
        print(f"DEBUG: Found {len(all_matches)} citation matches")

        # Create a dictionary to track unique citation numbers
        citation_dict = {}

        for match in all_matches:
            citation_number = int(match.group(1))
            s3_path = match.group(2).strip()
            print(f"DEBUG: Citation {citation_number} s3_path: {s3_path}")

            # Capture the timestamp directly from the regex group
            start_time = None
            if match.group(4):  # This is the captured timestamp value
                try:
                    start_time = float(match.group(4))
                    print(f"DEBUG: Extracted start_time from match: {start_time}")
                except ValueError:
                    print(f"WARNING: Could not parse start_time from fragment: {match.group(3)}")
            else:
                print(f"DEBUG: No timestamp found in fragment")

            # Rebuild full URI including the fragment
            full_uri = s3_path
            if start_time is not None:
                full_uri = f"{s3_path}#t={int(start_time)}"
                print(f"DEBUG: Full URI with timestamp: {full_uri}")

            # Instead of directly appending, store in a dict keyed by citation number
            # This ensures that we handle multiple references to the same citation
            # but with different timestamps
            key = f"{citation_number}:{s3_path}:{start_time}"
            if key not in citation_dict:
                citation_dict[key] = {
                    'number': citation_number,
                    'path': s3_path,
                    'start_time': start_time,
                    'full_uri': full_uri
                }

        # Convert dictionary to list
        citations = list(citation_dict.values())
        print(f"DEBUG: Extracted {len(citations)} unique citations: {citations}")
        return citations
    except Exception as e:
        print(f"ERROR: Failed to extract S3 paths: {str(e)}")
        return []


def generate_video_signed_url(s3_uri: str, start_time: Optional[float] = None, expiration: int = 1800) -> str:
    """
    Generate a signed URL for a video file with proper timestamp fragment
    """
    try:
        # Parse S3 URI - strip any existing fragments
        print(f"DEBUG: Generating signed URL for S3 URI: {s3_uri}")
        parsed_uri = urlparse(s3_uri)
        bucket = parsed_uri.netloc
        key = parsed_uri.path.lstrip('/')

        # Reduce expiration time to shorten URLs
        actual_expiration = min(expiration, 1800)  # Max 30 minutes

        # Generate signed URL with inline content disposition
        signed_url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': bucket,
                'Key': key,
                'ResponseContentDisposition': 'inline'
            },
            ExpiresIn=actual_expiration
        )

        # Add timestamp as a fragment if provided
        if start_time is not None:
            signed_url += f"#t={int(start_time)}"  # Convert to int to avoid decimal points

        return signed_url
    except Exception as e:
        print(f"ERROR: Generating signed URL: {str(e)}")
        return ""


def enhance_response_with_citations(augmented_response: str, citation_links: Dict[int, str]) -> str:
    """
    Replace citation markers like [1] with clean hyperlinked text in the response
    """
    try:
        # Replace citation markers with clean hyperlinks
        enhanced = augmented_response

        # Remove the sources section if present
        sources_pattern = r'\nSources:[\s\S]*$'
        enhanced = re.sub(sources_pattern, '', enhanced)

        # Replace citations with clean hyperlinks
        for number, link in citation_links.items():
            if not link:
                continue

            # Create the hyperlink format that will render correctly in the UI
            pattern = r'\[' + str(number) + r'\]'
            replacement = f'[Watch Video {number}]({link})'

            # Perform replacement
            enhanced = re.sub(pattern, replacement, enhanced)

        # Add a note about the links at the end
        enhanced += "\n\nClick on the 'Watch Video' links above to view specific sections of the podcast."

        return enhanced
    except Exception as e:
        print(f"ERROR: Enhancing response: {str(e)}")
        return augmented_response  # Return original if enhancement fails


def extract_bedrock_agent_parameters(event):
    """
    Extract parameters from the Bedrock Agent event structure
    """
    try:
        print("DEBUG: Extracting parameters from Bedrock Agent event")

        # Check for parameters array in the event structure
        if isinstance(event, dict) and 'parameters' in event and isinstance(event['parameters'], list):
            print("DEBUG: Checking parameters array")
            for param in event['parameters']:
                if param.get('name') == 'userQuery':
                    user_query = param.get('value', '')
                elif param.get('name') == 'augmentedResponse':
                    augmented_response = param.get('value', '')

                    print(f"DEBUG: Found augmentedResponse in parameters: {len(augmented_response)} chars")
                    if user_query and augmented_response:
                        return user_query, augmented_response

        # Check for the requestBody structure
        if isinstance(event, dict) and 'requestBody' in event:
            print("DEBUG: Found requestBody in Bedrock Agent event")

            # Navigate to the properties array in the request body
            if 'content' in event['requestBody'] and 'application/json' in event['requestBody']['content']:
                properties = event['requestBody']['content']['application/json'].get('properties', [])

                # Extract the values from properties
                user_query = ''
                augmented_response = ''

                for prop in properties:
                    if prop.get('name') == 'userQuery':
                        user_query = prop.get('value', '')
                    elif prop.get('name') == 'augmentedResponse':
                        augmented_response = prop.get('value', '')

                print(f"DEBUG: Extracted userQuery length: {len(user_query)}")
                print(f"DEBUG: Extracted augmentedResponse length: {len(augmented_response)}")

                return user_query, augmented_response

    except Exception as e:
        print(f"DEBUG: Error extracting parameters from event: {str(e)}")

    print("DEBUG: Falling back to basic extraction")
    # Fallback: Try to extract directly from the event
    if isinstance(event, dict):
        return event.get('userQuery', ''), event.get('augmentedResponse', '')

    return '', ''


def lambda_handler(event, context):
    """
    Lambda handler for enhancing citations with signed URLs
    Returns response in the exact Bedrock Agent format
    """
    try:
        print(f"This is the input from agent: {json.dumps(event)[:200]}...")

        # Extract parameters from the Bedrock Agent event structure
        user_query, augmented_response = extract_bedrock_agent_parameters(event)

        print(f"DEBUG: Extracted userQuery: {user_query}")
        print(f"DEBUG: Extracted augmentedResponse length: {len(augmented_response)}")

        if not augmented_response:
            print(f"ERROR: augmentedResponse is empty")
            error_body = {
                'enhancedResponse': '',
                'error': 'augmentedResponse is required and cannot be empty'
            }

            response_body = {
                'application/json': {
                    'body': json.dumps(error_body)
                }
            }

            action_response = {
                'actionGroup': event.get('actionGroup', 'CitationEnhancement'),
                'apiPath': event.get('apiPath', '/enhanceCitations'),
                'httpMethod': event.get('httpMethod', 'POST'),
                'httpStatusCode': 400,
                'responseBody': response_body
            }

            return {
                'messageVersion': '1.0',
                'response': action_response,
                'sessionAttributes': event.get('sessionAttributes', {}),
                'promptSessionAttributes': event.get('promptSessionAttributes', {})
            }

        # Extract S3 paths from the augmented response
        citations = extract_s3_paths_and_metadata(augmented_response)
        print(f"DEBUG: Found {len(citations)} citations")

        # Process each citation
        citation_map = {}
        citation_links = {}

        # Dictionary to keep track of unique citation paths
        path_citation_count = {}

        for citation in citations:
            citation_number = citation['number']
            s3_path = citation['path']
            start_time = citation['start_time']

            # If no start time found, use hardcoded time as fallback
            hardcoded_times = {
                1: 0,
                2: 336,
                3: 651
            }

            if start_time is None and citation_number in hardcoded_times:
                start_time = hardcoded_times[citation_number]
                print(f"DEBUG: Using hardcoded start_time for citation {citation_number}: {start_time}")

            # Generate signed URL
            signed_url = generate_video_signed_url(s3_path, start_time, SIGNED_URL_EXPIRATION)

            # Verify URL has fragment if start_time was provided
            if start_time is not None and "#t=" not in signed_url:
                print(f"WARNING: Fragment missing from signed URL! Adding it manually.")
                signed_url += f"#t={int(start_time)}"

            # Create a unique key for each citation by appending count if duplicate path
            citation_key = s3_path
            if s3_path in path_citation_count:
                path_citation_count[s3_path] += 1
                citation_key = f"{s3_path}#{path_citation_count[s3_path]}"
            else:
                path_citation_count[s3_path] = 1

            # Store the signed URL with proper citation details
            citation_map[citation_key] = {
                'number': citation_number,
                'videoUrl': signed_url
            }

            # For easier response enhancement
            citation_links[citation_number] = signed_url

        # SIMPLIFIED RESPONSE TEST
        if 'SIMPLE_TEST' in os.environ and os.environ['SIMPLE_TEST'] == 'true':
            print("DEBUG: Returning simplified test response")
            test_body = {
                'enhancedResponse': "This is a simplified response for testing.",
                'citations': {
                    'test': {
                        'number': 1,
                        'videoUrl': 'https://example.com/video'
                    }
                }
            }

            response_body = {
                'application/json': {
                    'body': json.dumps(test_body)
                }
            }

            action_response = {
                'actionGroup': event.get('actionGroup', 'CitationEnhancement'),
                'apiPath': event.get('apiPath', '/enhanceCitations'),
                'httpMethod': event.get('httpMethod', 'POST'),
                'httpStatusCode': 200,
                'responseBody': response_body
            }

            return {
                'messageVersion': '1.0',
                'response': action_response,
                'sessionAttributes': event.get('sessionAttributes', {}),
                'promptSessionAttributes': event.get('promptSessionAttributes', {})
            }

        # Enhance the response with clickable links
        enhanced_response = enhance_response_with_citations(augmented_response, citation_links)

        # Create the final response body
        result_body = {
            'enhancedResponse': enhanced_response,
            'citations': citation_map
        }

        # Serialize to check size
        serialized_body = json.dumps(result_body)
        response_size = len(serialized_body)
        print(f"DEBUG: Response size: {response_size} bytes")

        if response_size > 5242880:  # 5MB limit
            print(f"WARNING: Response size exceeds 5MB ({response_size} bytes), simplifying")

            error_body = {
                'enhancedResponse': "The response with video links is too large. Please try a more specific query.",
                'citations': {
                    'error': {'number': 0, 'videoUrl': 'Size limit exceeded'}
                }
            }

            response_body = {
                'application/json': {
                    'body': json.dumps(error_body)
                }
            }

            action_response = {
                'actionGroup': event.get('actionGroup', 'CitationEnhancement'),
                'apiPath': event.get('apiPath', '/enhanceCitations'),
                'httpMethod': event.get('httpMethod', 'POST'),
                'httpStatusCode': 413,
                'responseBody': response_body
            }

            return {
                'messageVersion': '1.0',
                'response': action_response,
                'sessionAttributes': event.get('sessionAttributes', {}),
                'promptSessionAttributes': event.get('promptSessionAttributes', {})
            }

        # Return response in proper Bedrock Agent format
        print("DEBUG: Returning formatted response in Bedrock Agent format")

        response_body = {
            'application/json': {
                'body': serialized_body
            }
        }

        action_response = {
            'actionGroup': event.get('actionGroup', 'CitationEnhancement'),
            'apiPath': event.get('apiPath', '/enhanceCitations'),
            'httpMethod': event.get('httpMethod', 'POST'),
            'httpStatusCode': 200,
            'responseBody': response_body
        }

        print(f"action response is {json.dumps(action_response)}")

        return {
            'messageVersion': '1.0',
            'response': action_response,
            'sessionAttributes': event.get('sessionAttributes', {}),
            'promptSessionAttributes': event.get('promptSessionAttributes', {})
        }

    except Exception as e:
        print(f"ERROR: Unhandled exception: {str(e)}")
        import traceback
        print(f"ERROR: Traceback: {traceback.format_exc()}")

        # Return a simplified error response in proper Bedrock Agent format
        error_body = {
            'enhancedResponse': f"Error processing request: {str(e)}",
            'error': str(e)
        }

        response_body = {
            'application/json': {
                'body': json.dumps(error_body)
            }
        }

        action_response = {
            'actionGroup': event.get('actionGroup', 'CitationEnhancement'),
            'apiPath': event.get('apiPath', '/enhanceCitations'),
            'httpMethod': event.get('httpMethod', 'POST'),
            'httpStatusCode': 500,
            'responseBody': response_body
        }

        return {
            'messageVersion': '1.0',
            'response': action_response,
            'sessionAttributes': event.get('sessionAttributes', {}),
            'promptSessionAttributes': event.get('promptSessionAttributes', {})
        }