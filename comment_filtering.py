import pandas as pd
from pathlib import Path # Import the Path object to handle file paths correctly

# --- 1. Load Data ---
file_path = 'youtube_comments_ZCngKo4zBH8.csv'
try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {len(df)} comments.")
    print("--- Data Sample ---")
    print(df.head())
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the file name and location.")
    exit() # Exit the script if the file is not found

# --- 2. Apply Rule-Based Spam Filtering ---
# This main 'if' block will run only if the 'text' column exists in the data
if 'text' in df.columns:
    # Ensure the text column is treated as a string to prevent errors
    df['text'] = df['text'].astype(str)

    # Rule 1: Check for URLs using a regular expression
    df['contains_url'] = df['text'].str.contains(r'http|www\.|\.com|\.net|\.org|\.co|\.kr|bit\.ly', case=False, na=False)

    # Rule 2: Check for common spam keywords
    spam_keywords = ['subscribe', 'free', 'event', 'giveaway', 'win', 'promo', 'crypto', 'lotto', 'winner', 'check out my channel']
    keyword_pattern = '|'.join(spam_keywords)
    df['contains_spam_keyword'] = df['text'].str.contains(keyword_pattern, case=False, na=False)

    # Rule 3: Check for very short comments
    df['is_too_short'] = df['text'].str.len() <= 3

    # Rule 4: Check for spam IDs
    spam_ids = ['19금', '19', 'Free']  # Add your spam IDs here
    df['contains_spam_id'] = df['author'].isin(spam_ids)


    # Combine rules: A comment is spam if any of the rules are True
    df['is_spam_by_rule'] = df['contains_url'] | df['contains_spam_keyword'] | df['is_too_short']

    # --- 3. Review and Save the Results ---
    # Filter the DataFrame to get only the comments flagged as spam
    spam_comments = df[df['is_spam_by_rule'] == True]
    spam_count = len(spam_comments)

    print("\n--- [Rule-Based Filtering Results] ---")
    print(f"A total of {spam_count} comments have been flagged as potential spam.")

    # Display a sample of the flagged comments
    print("\n--- Sample of Flagged Spam Comments ---")
    print(spam_comments[['author', 'text']].head(10))

    # --- 4. Save the flagged comments to a specific folder ---
    # Define the target path using the home directory symbol '~'
    target_path_str = '~/Documents/Workspace/project/ai_finder/flagged_spam_comments.csv'
    
    # Create a Path object and expand '~' to the full user home directory path
    full_path = Path(target_path_str).expanduser()
    
    # Create the target directory if it doesn't exist. This prevents errors.
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the 'spam_comments' DataFrame to the specified path
    spam_comments.to_csv(full_path, index=False, encoding='utf-8-sig')
    
    print(f"\nSuccessfully saved the flagged spam comments to:")
    print(full_path)

# This 'else' block runs only if the 'text' column was not found in the first place.
else:
    print("\nError: A 'text' column was not found in the DataFrame. Cannot perform spam filtering.")