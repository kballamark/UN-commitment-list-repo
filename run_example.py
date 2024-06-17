import utils.TableManipulation as TM
import utils.GPTTextAnalyzer as GPTTA

# Create an instance of table manipulator class 
table_manipulator = TM.TableManipulation()

# Use the instance to call the read_excel_file method
original_table = table_manipulator.read_excel_file('commitments_list.xlsx')

# Initialize the analyzer with an OpenAI API key
api_key = "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"

# Create an instance of GPTTextAnalyzer class
analyzer = GPTTA.GPTTextAnalyzer(api_key)

# Evaluate financial viability 
with open('prompts/financial_viability.txt', 'r') as file:
    financial_viability_prompt = file.read()

evaluated_table_financial = analyzer.evaluate(original_table.head(6), ['Resources', 'Description'], financial_viability_prompt, 'Evaluation_financial')

print(evaluated_table_financial['Evaluation_financial'])

# Evaluate accountability
with open('prompts/accountability.txt', 'r') as file:
    accountability_prompt = file.read()

evaluated_table_accountability = analyzer.evaluate(original_table[424:427], ['Deliverables', 'Description', 'Expected Impact'], accountability_prompt, 'Evaluation_accountability')

print(evaluated_table_accountability['Evaluation_accountability'])

print('Successful run')