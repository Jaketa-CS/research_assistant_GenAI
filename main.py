import logging  # Used to record events and errors during the execution of a program
import arxivscraper
import pandas as pd
import json
from openai import OpenAI
import time
import os

# Set up logging
logging.basicConfig(level=logging.INFO)  # Configure the logging module to record INFO-level messages and above
logger = logging.getLogger(__name__)  # Create a logger object for the current module

api_key = os.getenv("OPENAI_API_KEY")
logger.info("THIS_IS_THE_KEY: %s", api_key)

client = OpenAI(api_key=api_key)


def scrape_ai(start_date, end_date, category='cs.AI'):
    """
    Scrape arXiv data based on the specified date range and category.
    Save the data to a JSON file.

    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        category (str): arXiv category.

    Returns:
        None
    """
    folder = "ARXIV"
    if not os.path.exists(folder):
        os.makedirs(folder)

    try:
        scraper = arxivscraper.Scraper(category=category, date_from=start_date, date_until=end_date,
                                       filters={'categories': ['cs.AI']})
        output = scraper.scrape()

        if not output:
            logger.warning("No data retrieved from the arXiv scraper.")
            return

        cols = ('id', 'title', 'abstract', 'doi', 'created', 'url', 'authors')
        df = pd.DataFrame(output, columns=cols)
        json_data = df.to_json(orient='records')
        formatted_json = json.loads(json_data)
        with open('ARXIV/arxiv_data.json', 'w') as file:
            json.dump(formatted_json, file, indent=4)
    except Exception as e:
        logger.error("Error during scraping: %s", str(e))



def upload_file(assistant_id, folder='ARXIV'):
    """
    Upload files from the specified folder to OpenAI and associate them with the assistant.

    Args:
        assistant_id (str): Assistant ID.
        folder (str): Folder containing files to upload.

    Returns:
        list: List of file IDs.
    """
    file_ids = []

    try:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)

            if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
                response = client.files.create(
                    file=open(file_path, "rb"),
                    purpose="assistants"
                )

                file_id = response.id

                if file_id:
                    assistant_file = client.beta.assistants.files.create(
                        assistant_id=assistant_id,
                        file_id=file_id
                    )
                    file_ids.append(file_id)
    except Exception as e:
        logger.error("Error during file upload: %s", str(e))

    return file_ids


def setup_assistant(client, assistant_name, model="gpt-3.5-turbo-1106"):
    """
    Create a new assistant.

    Args:
        client: OpenAI client.
        assistant_name (str): Name for the assistant.
        model (str): OpenAI model to use.

    Returns:
        tuple: Assistant ID and Thread ID.
    """
    try:
        assistant = client.beta.assistants.create(
            name=assistant_name,
            instructions=f"""
                You are an intelligent and helpful research assistant. Your name is {assistant_name}. You will work with the user to
                help them learn new updates on AI advancements from data within a json file. You will analyze the json file,
                find the most relevant papers to the user's learning request, and output a summary of the articles and their importance in one message. 
                Always output the links to the papers after you summarize them.
            """,
            model=model,
            tools=[{"type": "retrieval"}, {"type": "code_interpreter"}]
        )
        thread = client.beta.threads.create()
        return assistant.id, thread.id
    except Exception as e:
        logger.error("Error during assistant setup: %s", str(e))
        return None, None


def send_message(client, thread_id, task, file_ids):
    """
    Send a message to the assistant.

    Args:
        client: OpenAI client.
        thread_id (str): Thread ID.
        task (str): Task content.
        file_ids (list): List of file IDs.

    Returns:
        dict: Thread message details.
    """
    try:
        thread_message = client.beta.threads.messages.create(
            thread_id,
            role="user",
            content=task,
            file_ids=file_ids
        )
        return thread_message
    except Exception as e:
        logger.error("Error during sending message: %s", str(e))
        return None


def run_assistant(client, assistant_id, thread_id):
    """
    Run the assistant.

    Args:
        client: OpenAI client.
        assistant_id (str): Assistant ID.
        thread_id (str): Thread ID.

    Returns:
        list: List of thread messages.
    """
    try:
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id
        )

        while run.status == "in_progress" or run.status == "queued":
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )

        if run.status == "completed":
            return client.beta.threads.messages.list(
                thread_id=thread_id
            )
    except Exception as e:
        logger.error("Error during assistant run: %s", str(e))
        return None


def save_session(assistant_id, thread_id, user_name_input, file_ids, file_path='arxiv_sessions.json'):
    """
    Save the current session details to a file.

    Args:
        assistant_id (str): Assistant ID.
        thread_id (str): Thread ID.
        user_name_input (str): User name input.
        file_ids (list): List of file IDs.
        file_path (str): File path.

    Returns:
        None
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)
        else:
            data = {"sessions": {}}

        next_session_number = str(len(data["sessions"]) + 1)

        data["sessions"][next_session_number] = {
            "Assistant ID": assistant_id,
            "Thread ID": thread_id,
            "User Name Input": user_name_input,
            "File IDs": file_ids
        }

        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        logger.error("Error during saving session: %s", str(e))


def display_sessions(file_path='arxiv_sessions.json'):
    """
    Display available sessions.

    Args:
        file_path (str): File path.

    Returns:
        None
    """
    try:
        if not os.path.exists(file_path):
            print("No sessions available.")
            return

        with open(file_path, 'r') as file:
            data = json.load(file)

        print("Available Sessions:")
        for number, session in data["sessions"].items():
            print(f"Session {number}: {session['User Name Input']}")
    except Exception as e:
        logger.error("Error during displaying sessions: %s", str(e))


def get_session_data(session_number, file_path='arxiv_sessions.json'):
    """
    Get session data based on the session number.

    Args:
        session_number (str): Session number.
        file_path (str): File path.

    Returns:
        tuple: Assistant ID, Thread ID, User Name Input, and File IDs.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

        session = data["sessions"].get(session_number)
        if session:
            return session["Assistant ID"], session["Thread ID"], session["User Name Input"], session["File IDs"]
        else:
            print("Session not found.")
            return None, None
    except Exception as e:
        logger.error("Error during getting session data: %s", str(e))
        return None, None


def collect_message_history(assistant_id, thread_id, user_name_input):
    """
    Collect and save message history to a file.

    Args:
        assistant_id (str): Assistant ID.
        thread_id (str): Thread ID.
        user_name_input (str): User name input.

    Returns:
        str: Message log information.
    """
    try:
        messages = run_assistant(client, assistant_id, thread_id)
        message_dict = json.loads(messages.model_dump_json())

        with open(f'{user_name_input}_message_log.txt', 'w') as message_log:
            for message in reversed(message_dict['data']):
                text_value = message['content'][0]['text']['value']

                if message['role'] == 'assistant':
                    prefix = f"{user_name_input}: "
                else:
                    prefix = "You: "

                message_log.write(prefix + text_value + '\n')

        return f"Messages saved to {user_name_input}_message_log.txt"
    except Exception as e:
        logger.error("Error during collecting message history: %s", str(e))
        return "Error collecting message history."


def main_loop():
    try:
        print("\n------------------------------ Welcome to Arxiv GPT! ------------------------------\n")
        user_choice = input("Type 'n' to make a new agent. Press 'Enter' to choose an existing session. ")
        if user_choice == 'n':
            scrape_ai(start_date='2023-12-16', end_date='2023-12-16')  # Adjust to today's date
            user_name_input = input("Type a Name for this Assistant (usually today's date is best): ")
            IDS = setup_assistant(client, assistant_name=user_name_input)
            assistant_id, thread_id, file_ids = IDS[0], IDS[1], []
            file_ids.extend(upload_file(assistant_id))
            save_session(assistant_id, thread_id, user_name_input, file_ids)
            logger.info(f"Created Session with {user_name_input}, Assistant ID: {assistant_id} and Thread ID: {thread_id}\n"
                        f"Please tell the assistant what specific subject you want to focus on.")
        else:
            display_sessions()
            chosen_session_number = input("Enter the session number to load: ")
            assistant_id, thread_id, user_name_input, file_ids = get_session_data(chosen_session_number)
            logger.info(f"Started a new session with {user_name_input}, Assistant ID: {assistant_id} and Thread ID: {thread_id}")
        if assistant_id and thread_id:
            while True:
                user_message = input("You: ")
                if user_message.lower() in {'exit', 'exit.'}:
                    print("Exiting the program.")
                    print(collect_message_history(assistant_id, thread_id, user_name_input))
                    break
                send_message(client, thread_id, user_message, file_ids)
                messages = run_assistant(client, assistant_id, thread_id)
                message_dict = json.loads(messages.model_dump_json())
                most_recent_message = message_dict['data'][0]
                assistant_message = most_recent_message['content'][0]['text']['value']
                print(f"{user_name_input}: {assistant_message}")
    except Exception as e:
        logger.error("Error in main loop: %s", str(e))


if __name__ == "__main__":
    main_loop()
