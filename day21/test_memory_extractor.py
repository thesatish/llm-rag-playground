from memory_extractor import extract_memory


memory = extract_memory(
    "What is the current time?",
    "The current time is 10 PM."

)

print("Extracted Memory:")
print(memory)
