from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from groq import Groq
import json
import os

load_dotenv()

client = Groq(
    api_key = os.environ.get("GROQ_API_KEY"),
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or specify your frontend URL)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Text(BaseModel):
    story_text: str

class Shot(BaseModel):
    shot_no: int = Field(gt=0)
    camera_angle: str
    lens: str
    description: str 

class StoryBoard(BaseModel):
    message: str
    final_story: str
    shot_list: list[Shot]

@app.get("/")
def home():
    return FileResponse("index.html")

@app.post("/generate-shots", response_model = StoryBoard)
def generate_shots(text: Text):
    clean_text = text.story_text.strip()

    if clean_text == "":
        raise HTTPException(
            status_code = 400,
            detail = "Input text is empty"
        )
    
    clean_text = clean_text.replace("\n", " ")

    prompt = f"""
        You are a professional screenwriter, commercial filmmaker, and cinematic storyboard artist.

        Your task is to take a short story idea and expand it into a complete cinematic story suitable for a film scene, advertisement, short film, or branded commercial.

        The generated story must:
        - feel emotionally engaging
        - feel visually cinematic
        - have a clear beginning, middle, and ending
        - contain realistic visual moments
        - be optimized for filmmaking and shot design
        - remain concise but immersive

        Then generate a professional cinematic shot list that visually tells the story.

        Generate the appropriate number of shots based on the scale and pacing of the story.

        For each shot include:
        - shot_no
        - camera_angle
        - lens
        - description

        SHOT RULES:
        - Use real filmmaking terminology
        - Use realistic camera language
        - Keep descriptions BRIEF and visual
        - Avoid overly long descriptions
        - Ensure shots logically progress through the story
        - Include emotional and cinematic visual framing
        - Use varied shot compositions

        IMPORTANT RULES:
        - Return ONLY valid JSON
        - Do NOT include markdown
        - Do NOT include explanations
        - Do NOT include notes
        - Do NOT include headings outside JSON
        - Do NOT write anything before or after the JSON
        - Ensure all property names use double quotes
        - Ensure JSON is parsable using json.loads()
        - Do NOT include trailing commas

        Return the response using this EXACT JSON structure:

        {{
            "message": "Successful",
            "final_story": "Complete cinematic story here",
            "shot_list": [
                {{
                    "shot_no": 1,
                    "camera_angle": "Wide Shot",
                    "lens": "35mm",
                    "description": "A lonely man walks through an abandoned railway station under flickering lights."
                }}
            ]
        }}

        STORY IDEA:
        {clean_text}
        """
        
    
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    model="llama-3.3-70b-versatile",
)
    response = chat_completion.choices[0].message.content

    response = response.replace("```json", "")
    response = response.replace("```", "")
    response = response.strip()

    try:
        new_response = json.loads(response)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code = 400,
            detail = "Response cannot be converted to JSON"
        )
    except Exception as e:
        return {"Error" : str(e)}
    
    shots = []

    for shot in new_response["shot_list"]:
        shot_info = Shot(
            shot_no = shot["shot_no"],
            camera_angle = shot["camera_angle"],
            lens = shot["lens"],
            description = shot["description"]
        )

        shots.append(shot_info)

    final_response = StoryBoard(
        message = new_response["message"],
        final_story = new_response["final_story"],
        shot_list = shots
    )

    return final_response