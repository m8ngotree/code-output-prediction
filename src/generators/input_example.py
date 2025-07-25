"""
Example usage of the InputGenerator class.

This script demonstrates how to use the InputGenerator to analyze
Python code and generate diverse test inputs.
"""

from input_generator import InputGenerator


def main():
    """Demonstrate InputGenerator usage."""
    
    # Initialize generator
    generator = InputGenerator()
    
    # Example 1: Simple function with typed parameters
    print("=" * 60)
    print("Example 1: Function with typed parameters")
    print("=" * 60)
    
    code1 = """
def calculate_area(length: int, width: int, name: str = "rectangle") -> float:
    '''Calculate area of a rectangle.'''
    if length <= 0 or width <= 0:
        raise ValueError("Dimensions must be positive")
    return length * width
"""
    
    result1 = generator.generate_and_save(code1, "example1_inputs.json")
    print(f"✅ Generated {result1['summary']['total_inputs']} inputs")
    print(f"📁 Saved to: {result1['output_file']}")
    print(f"📊 Requirements found: {result1['summary']['requirements_found']}")
    print(f"   - Normal cases: {result1['summary']['normal_cases']}")
    print(f"   - Edge cases: {result1['summary']['edge_cases']}")
    
    # Example 2: Function with input() calls
    print("\n" + "=" * 60)
    print("Example 2: Interactive input function")
    print("=" * 60)
    
    code2 = """
def interactive_calculator():
    '''Simple calculator with user input.'''
    operation = input("Enter operation (+, -, *, /): ")
    num1 = float(input("Enter first number: "))
    num2 = float(input("Enter second number: "))
    
    if operation == "+":
        return num1 + num2
    elif operation == "-":
        return num1 - num2
    elif operation == "*":
        return num1 * num2
    elif operation == "/":
        if num2 == 0:
            return "Error: Division by zero"
        return num1 / num2
    else:
        return "Error: Invalid operation"
"""
    
    result2 = generator.generate_and_save(code2, "example2_inputs.json")
    print(f"✅ Generated {result2['summary']['total_inputs']} inputs")
    print(f"📁 Saved to: {result2['output_file']}")
    
    # Example 3: File processing function
    print("\n" + "=" * 60)
    print("Example 3: File processing function")
    print("=" * 60)
    
    code3 = """
def process_file(filename: str, encoding: str = "utf-8"):
    '''Process a text file and return word count.'''
    try:
        with open(filename, 'r', encoding=encoding) as f:
            content = f.read()
        
        words = content.split()
        return {
            'filename': filename,
            'word_count': len(words),
            'char_count': len(content),
            'line_count': len(content.splitlines())
        }
    except FileNotFoundError:
        return {'error': f'File {filename} not found'}
"""
    
    result3 = generator.generate_and_save(code3, "example3_inputs.json")
    print(f"✅ Generated {result3['summary']['total_inputs']} inputs")
    print(f"📁 Saved to: {result3['output_file']}")
    print(f"🗂️  Temporary files created: {len(result3['inputs_data']['temp_files'])}")
    
    # Example 4: Complex data processing
    print("\n" + "=" * 60)
    print("Example 4: Complex data structures")
    print("=" * 60)
    
    code4 = """
def analyze_sales_data(sales_data: list, categories: dict, threshold: float = 100.0):
    '''Analyze sales data and return insights.'''
    if not sales_data:
        return {'error': 'No sales data provided'}
    
    total_sales = sum(item.get('amount', 0) for item in sales_data)
    high_value_sales = [item for item in sales_data 
                       if item.get('amount', 0) > threshold]
    
    category_totals = {}
    for item in sales_data:
        category = item.get('category', 'unknown')
        category_name = categories.get(category, category)
        category_totals[category_name] = category_totals.get(category_name, 0) + item.get('amount', 0)
    
    return {
        'total_sales': total_sales,
        'high_value_count': len(high_value_sales),
        'category_breakdown': category_totals,
        'average_sale': total_sales / len(sales_data) if sales_data else 0
    }
"""
    
    result4 = generator.generate_and_save(code4, "example4_inputs.json")
    print(f"✅ Generated {result4['summary']['total_inputs']} inputs")
    print(f"📁 Saved to: {result4['output_file']}")
    
    # Example 5: Show detailed analysis
    print("\n" + "=" * 60)
    print("Example 5: Detailed Analysis of Generated Inputs")
    print("=" * 60)
    
    # Analyze the last example in detail
    inputs_data = result4['inputs_data']
    
    print(f"📋 Requirements Analysis:")
    for i, req in enumerate(inputs_data['requirements'], 1):
        print(f"  {i}. {req['name']} ({req['param_type']})")
        print(f"     Type: {req['inferred_type']}")
        print(f"     Description: {req['description']}")
        print(f"     Inputs generated: {req['input_count']}")
    
    print(f"\n📊 Input Categories:")
    for category, inputs in inputs_data['inputs'].items():
        if inputs:
            print(f"  {category.title()} ({len(inputs)} inputs):")
            for inp in inputs[:2]:  # Show first 2 examples
                data_preview = str(inp['data'])[:50]
                if len(str(inp['data'])) > 50:
                    data_preview += "..."
                print(f"    - {inp['description']}: {data_preview}")
    
    # Cleanup
    print(f"\n🧹 Cleaning up temporary files...")
    generator.cleanup_temp_files()
    print("✅ Cleanup complete")
    
    print(f"\n🎉 Input generation examples completed!")
    print(f"📁 Check the data/inputs/ directory for generated files")


if __name__ == "__main__":
    main()