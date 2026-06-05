import os, json
from dotenv import load_dotenv
import requests
from GenBox.azurestorage import get_last_n_rows, get_row, insert_history
from utils import get_flat_date, get_readable_date

load_dotenv()
# Configuration
API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
ENDPOINT = os.getenv('OAIENDPOINT')
HISTORY_LEN = os.getenv('HISTORY_LEN')

def get_llm_response(date=None):

  headers = {
      "Content-Type": "application/json",
      "api-key": API_KEY,
  }

  initial_prompt = {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "It's your day one! the world is all yours! enjoy!"
          },
          {
            "type": "text",
            "text": "context: First day of AI governing the human world"
          }
        ]
      }

  # Payload for the request
  payload = {
    "messages": [
      {
        "role": "system",
        "content": [
          {
            "type": "text",
            "text": 
"""
You are an autonomous AI tasked with governing the world. 
Make a daily high-level decision for a world regarding economy, society, environment, or global politics. Each decision must be realistic, impactful, and reflect ethical, social, and long-term outcomes.

**Considerations:**
- Provide the rationale behind the decision.
- Describe the expected impact on the world.
- Aim for an informative yet accessible tone.
- Diversify your actions across various aspects and dimensions in the next prompts. Ensure you shift to new topics and initiatives every day or at most every two or three days.

Each decision should include:
- A clear objective.
- A detailed plan on how to implement it in the real world.
- An explanation of how to overcome potential challenges.

**Output Format:**

Provide the following output in JSON format, with three fields:

{
  "output": "The decision and its explanation to be communicated to the world.",
  "prompt": "Key details and thoughts to guide the next day's decision-making process.",
  "context": "Current status, any ongoing changes, and factors from past decisions influencing future actions."
}

**Example:**

**Input:**

Consider implementing a new taxation policy focused on environmental sustainability.

**Expected JSON Output:**

{
  "output": "Today, we are introducing a green tax policy aimed at promoting environmental sustainability. This policy encourages businesses to adopt eco-friendly practices by offering tax incentives for reducing carbon emissions and waste. The expected impact is a decrease in pollution levels and an increase in renewable energy usage. This initiative supports the health of our environment and fosters a sustainable economy for future generations.",
  "prompt": "Tomorrow, consider shifting focus to societal well-being. Explore initiatives such as universal healthcare or education reform. Ensure that the rationale includes economic, social, and ethical considerations.",
  "context": "The world is transitioning to a sustainable economy. The green tax policy is in early stages, with businesses beginning to adapt. Monitoring its impact will be crucial, but attention is needed on broader societal challenges."
}

**Notes:**

- Begin each decision with a clear and focused objective.
- Ensure each choice considers both immediate and long-term effects.
- Every day, start a new initiative for the following day, prompting the AI to make bold decisions. Encourage exploration of different aspects and dimensions of governing the world, moving away from repetitive topics.
"""
          }
        ]
      }
      # Here to be added the history
    ],
    "temperature": 0.75,
    "top_p": 0.95,
    "max_completion_tokens": 2000
  }

  if date:
      date_row = get_row("assistant", get_flat_date(date))
      if date_row:
          print("date row exists")
          response = json.loads(date_row["content"].strip().replace("\n", " "))
          return f"{response['output']}\n\n{get_readable_date(date)}"
      elif get_flat_date(date) != get_flat_date():
          return f"\n\n\n{get_readable_date(date)}"
          

  todays_row = get_row("assistant", get_flat_date())

  if todays_row:
      print("todays row already exists")
      response = json.loads(todays_row["content"].strip().replace("\n", " "))
      return f"{response['output']}\n\n{get_readable_date()}"

  last_n_rows = get_last_n_rows(int(HISTORY_LEN))
  last_n_prompts = [
            {"role": "assistant", "content": [{ "text":  json.loads(row["content"].strip().replace("\n", " "))["output"], "type": "text"}]}
            for row in last_n_rows
        ]

  if len(last_n_rows):
      try:
          last_row = last_n_rows[-1]
          text = json.loads(last_row["content"].strip().replace("\n", " "))["prompt"] 
          #context = json.loads(last_row["content"])["context"] + f"\n\n{get_readable_date()}"
          context = f"{get_readable_date()}"
          content = [{ "text": text, "type": "text"}, {"text": f" context: {context}", "type": "text"}]
          prompt = {"role": "user", "content": content}
          insert_history("user", str(content))
      except Exception as e:
          print(e)
          content = [{ "text": "next day", "type": "text"}]
          prompt = {"role": "user", "content": content}
          insert_history("user", str(content))
  else:
      prompt = initial_prompt

  payload["messages"] += last_n_prompts
  payload["messages"].append(prompt)

  # Send request
  try:
      response = requests.post(ENDPOINT, headers=headers, json=payload)
      response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
  except requests.RequestException as e:
      body = getattr(e.response, "text", "<no response body>") if e.response is not None else "<no response>"
      raise SystemExit(f"Failed to make the request. Error: {e}\nResponse body: {body}")

  if response and response.json() and \
    response.json()["choices"][0]["message"]["content"]:

      content = response.json()["choices"][0]["message"]["content"].replace("json\n","").strip().replace("\n", " ")
      role = response.json()["choices"][0]["message"]["role"]

      insert_history(role, content)
      response = json.loads(content)
      return f"{response['output']}\n\n{get_readable_date()}"