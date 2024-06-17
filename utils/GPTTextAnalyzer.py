import openai
import re

class GPTTextAnalyzer:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)

    def analyze_text_with_gpt(self, text, engine="gpt-3.5-turbo-0125", max_tokens=1000, temperature=0.4):
        """
        Analyzes the given text using the specified GPT model.
        
        Parameters:
        - text (str): The text to be analyzed.
        - engine (str): The GPT model to use for analysis. gpt-4-0125-preview
        - max_tokens (int): The maximum number of tokens to generate in the completion.
        - temperature (float): Controls the randomness of the output. Lower values make the model more deterministic.
        
        Returns:
        - str: The content of the GPT model's response.
        """
        response = self.client.chat.completions.create(
            model=engine,
            messages=[{"role": "user", "content": text}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        # Accessing the first completion's message content
        response_text = response.choices[0].message.content.strip()
        
        # Print the response to the terminal
        print("GPT Response:", response_text)
        return response_text        

    def classify(self, text, prompt):
        full_prompt = f"{prompt}\n\nDescription: {text}\n\nClassification:"
        classification = self.analyze_text_with_gpt(full_prompt)
        # Use regex to find the first number in the classification text
        match = re.search(r'\b\d+\b', classification)
        if match:
            classification_num = int(match.group())
            if classification_num in [1, 2, 3]:
                return classification_num
        # Return -1 if no valid number is found
        return -1

    def evaluate(self, df, columns, prompt, evaluation_column):
        # Create a new column by concatenating text from specified columns
        df['combined_text'] = df[columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

        # Apply classification on the combined text
        df.loc[:, evaluation_column] = df['combined_text'].apply(lambda x: self.classify(x, prompt))

        # Drop if not needed
        df.drop('combined_text', axis=1, inplace=True)
        return df
