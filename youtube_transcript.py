import re
from youtube_transcript_api import YouTubeTranscriptApi
from tiktoken import get_encoding
from openai import ChatCompletion
from tiktoken import TokenCount

def get_spr_from_youtube(video_id):
    # Fetch the transcript
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    
    # Concatenate all the parts of the transcript
    full_transcript = " ".join([part['text'] for part in transcript])
    
    # Initialize the tokenizer with the encoding for gpt-4
    encoding = get_encoding("gpt-4")
    
    # Split the transcript into chunks of no more than 3000 tokens
    chunks = []
    chunk = ""
    for word in full_transcript.split():
        if len(encoding.encode(chunk + " " + word)) <= 3000:
            chunk += " " + word
        else:
            chunks.append(chunk.strip())
            chunk = word
    chunks.append(chunk.strip())
    
    # Convert each chunk into an SPR
    sprs = []
    for chunk in chunks:
        # Prepare the conversation
        conversation = [
            {"role": "system", "content": open_file('system_spr_encoder.txt')},
            {"role": "user", "content": chunk}
        ]
        
        # Generate the SPR
        response = OpenAI.ChatCompletion.create(model="gpt-4", messages=conversation)
        sprs.append(response['choices'][0]['message']['content'])
    
    return sprs
