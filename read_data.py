# Listing 2.1 Reading in a short story as text sample into Python
with open("the-verdict.txt", "r", encoding="utf-8") as f:
		raw_text = f.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99])
