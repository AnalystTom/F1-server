#!/usr/bin/env python3
"""
Quick test script for FastF1-Agent functionality
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import GeminiAgent, FASTF1_SYSTEM_INSTRUCTION

def test_code_execution():
    """Test the code execution functionality"""
    print("Testing code execution...")
    
    # Create a simple agent for testing
    agent = GeminiAgent(
        api_key="test-key",  # Won't be used for code execution test
        system_instruction="Test"
    )
    
    # Test simple Python code
    test_code = """
print("Hello FastF1-Agent!")
import pandas as pd
df = pd.DataFrame({'driver': ['HAM', 'VER'], 'time': [90.5, 91.2]})
print(df.to_string(index=False))
"""
    
    result = agent.execute_code(test_code)
    print(f"Return code: {result['return_code']}")
    print(f"Stdout: {result['stdout']}")
    if result['stderr']:
        print(f"Stderr: {result['stderr']}")
    
    return result['return_code'] == 0

def test_react_processing():
    """Test ReAct response processing with 2025 Canada GP query"""
    print("\nTesting ReAct processing with future F1 data query...")
    
    agent = GeminiAgent(
        api_key="test-key",
        system_instruction=FASTF1_SYSTEM_INSTRUCTION
    )
    
    # Mock response for 2025 Canada GP query (testing real FastF1 data access)
    mock_response = """### Thought
1. The user is asking for the fastest lap from the 2025 Canadian Grand Prix
2. I need to load the race session data for Canada 2025
3. Find the fastest lap across all drivers
4. Display the driver and lap time information

### Code
```python
import fastf1
import pandas as pd
import matplotlib.pyplot as plt
fastf1.Cache.enable_cache("/tmp/fastf1_cache")

# Load the 2025 Canadian Grand Prix race session
session = fastf1.get_session(2025, "Canada", "R")
session.load()

# Get the fastest lap from the race
fastest_lap = session.laps.pick_fastest()
driver_info = session.get_driver(fastest_lap['Driver'])

print(f"Fastest lap in 2025 Canadian Grand Prix:")
print(f"Driver: {driver_info['Abbreviation']} ({driver_info['FullName']})")
print(f"Lap Time: {fastest_lap['LapTime']}")
print(f"Lap Number: {fastest_lap['LapNumber']}")
```
<EXECUTE>

### Answer
The fastest lap data has been retrieved from the 2025 Canadian Grand Prix.
"""
    
    processed = agent._process_react_response(mock_response)
    print("Processed response:")
    print(processed)
    
    return "<EXECUTE>" in processed and "### Result" in processed

if __name__ == "__main__":
    print("FastF1-Agent Test Suite")
    print("=" * 30)
    
    # Test 1: Code execution
    execution_success = test_code_execution()
    
    # Test 2: ReAct processing
    react_success = test_react_processing()
    
    print("\n" + "=" * 30)
    print("Test Results:")
    print(f"Code Execution: {'✓ PASS' if execution_success else '✗ FAIL'}")
    print(f"ReAct Processing: {'✓ PASS' if react_success else '✗ FAIL'}")
    
    if execution_success and react_success:
        print("\n🎉 All tests passed! FastF1-Agent is ready.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)