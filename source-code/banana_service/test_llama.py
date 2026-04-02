from banana_service.llm import LocalLlamaLLM

def main():
    llm = LocalLlamaLLM()

    prompt = """
You are a financial analyst.

What was Apple’s closing price on February 27, 2026?
"""

    response = llm.generate(prompt, temperature=0.0)

    print("\n===== RESPONSE =====\n")
    print(response)


if __name__ == "__main__":
    main()
