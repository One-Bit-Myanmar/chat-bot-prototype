def prepare_conversation_pairs(file_path):
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utterances = [u.strip() for u in line.split("__eou__") if u.strip()]
            for i in range(len(utterances) - 1):
                speaker_a = f"{utterances[i]}"
                speaker_b = f"{utterances[i + 1]}"
                pairs.append((speaker_a, speaker_b))
    return pairs

def save_pairs_to_text(pairs, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for input_text, response_text in pairs:
            f.write(f"Input: {input_text}\n")
            f.write(f"Response: {response_text}\n\n")



# Usage
pairs = prepare_conversation_pairs("dataset/dialogues_validation.txt")
save_pairs_to_text(pairs, 'dataset/conver_val.txt')
print(f"Saved {len(pairs)} pairs to 'dataset/conver_val.txt'")

# Preview some pairs
for i in range(5):
    print("Input:", pairs[i][0])
    print("Response:", pairs[i][1])
    print()
