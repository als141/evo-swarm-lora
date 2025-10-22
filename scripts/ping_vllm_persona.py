import argparse

from openai import OpenAI


def query_persona(base_url: str, persona: str, prompt: str) -> str:
    client = OpenAI(base_url=base_url, api_key="EMPTY")
    response = client.chat.completions.create(
        model=f"Qwen/Qwen3-4B-Instruct-2507:{persona}",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="Ping a vLLM service exposing Multi-LoRA personas.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="Base URL of the vLLM OpenAI-compatible server.")
    parser.add_argument("--persona", default="persona_a", help="Persona adapter name.")
    parser.add_argument("--prompt", default="次の数列の一般項は？ 2,4,8,16,...", help="Prompt to send.")
    args = parser.parse_args()

    result = query_persona(args.base_url, args.persona, args.prompt)
    print(result)


if __name__ == "__main__":
    main()
