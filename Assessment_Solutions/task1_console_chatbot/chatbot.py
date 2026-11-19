"""
Task 1: Console based chat bot.

Algorithm:
    Step 1: get the input from the user
    Step 2: prepare the message object
    Step 3: invoke the LLM
    Step 4: print it in the console [question and answer]
    Step 5: quit
"""

from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic()
model = "claude-haiku-5-5"

SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and concisely."
EXIT_COMMANDS = {"quit", "exit", "bye", "q"}


def add_user_message(messages, text):
    messages.append({"role": "user", "content": text})


def add_assistant_message(messages, text):
    messages.append({"role": "assistant", "content": text})


def chat(messages):
    response = client.messages.create(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return "".join(block.text for block in response.content if block.type == "text")


def main():
    messages = []
    print("=" * 60)
    print(" Console Chat Bot (type 'quit' or 'exit' to end)")
    print("=" * 60)

    while True:
        # Step 1: get the input from the user
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not user_input:
            continue

        # Step 5: quit
        if user_input.lower() in EXIT_COMMANDS:
            break

        # Step 2: prepare the message object (keeps the conversation history)
        add_user_message(messages, user_input)

        # Step 3: invoke the LLM
        try:
            answer = chat(messages)
        except Exception as e:
            print(f"\n[Error calling the LLM: {e}]")
            messages.pop()  # drop the unanswered question
            continue

        add_assistant_message(messages, answer)

        # Step 4: print it in the console [question and answer]
        print(f"\nQuestion : {user_input}")
        print(f"Answer   : {answer}")

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
