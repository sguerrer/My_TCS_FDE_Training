# Task 1 – Console Chat Bot

A console chat bot that reads the user's input, sends it to Claude and prints the
question and answer as a conversation. The conversation history is kept, so the bot
remembers earlier turns.

## Algorithm
1. Get the input from the user
2. Prepare the message object
3. Invoke the LLM (Claude via the Anthropic API)
4. Print the question and answer in the console
5. Quit (`quit`, `exit`, `bye`, `q` or Ctrl+C)

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env   # then put your ANTHROPIC_API_KEY in .env
python chatbot.py
```

## Example
```
You: What is RAG?

Question : What is RAG?
Answer   : RAG (Retrieval-Augmented Generation) is ...

You: quit

Goodbye!
```
