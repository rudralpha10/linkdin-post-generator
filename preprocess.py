import json
import re
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# === 1. Setup LLM ===
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key="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"  # Replace with your actual API key
)

# === 2. Prompt Templates ===
extract_prompt = PromptTemplate.from_template("""
You are a helpful assistant that extracts metadata from LinkedIn posts.
Extract the following information:
- Topics or tags (e.g. AI, internships, careers)
- Audience (e.g. Students, Jobseekers, Founders)
Return the result as a JSON dictionary like this:
{{"tags": [...], "audience": "..."}}.

Post: {post}
""")

unify_prompt = PromptTemplate.from_template("""
You are a smart assistant that unifies similar tags into a consistent tag set.

Input tags: {tags}

Return a dictionary mapping original tags to unified ones.
Example: {{"AI": "Artificial Intelligence", "ML": "Machine Learning"}}
""")

# === 3. Helper: Clean Broken Unicode ===
def clean_unicode(text):
    """Remove invalid surrogate characters (broken emojis)"""
    return re.sub(r'[\ud800-\udfff]', '', text) if isinstance(text, str) else text

# === 4. Metadata Extraction ===
def extract_metadata(post_text):
    chain = extract_prompt | llm | StrOutputParser()
    response = chain.invoke({"post": post_text})
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        print("Invalid JSON response from model:", response)
        return {"tags": [], "audience": ""}

# === 5. Unify Similar Tags ===
def get_unified_tags(posts):
    all_tags = set(tag for post in posts for tag in post.get("tags", []))
    unique_tags_list = list(all_tags)
    chain = unify_prompt | llm | StrOutputParser()

    response = chain.invoke({"tags": unique_tags_list})
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        print("Failed to unify tags. Response:", response)
        return {tag: tag for tag in unique_tags_list}  # fallback

# === 6. Main Processing Function ===
def process_posts(raw_file_path, processed_file_path):
    with open(raw_file_path, encoding='utf-8') as file:
        posts = json.load(file)

    enriched_posts = []
    for post in posts:
        try:
            cleaned_text = clean_unicode(post['text'])
            post['text'] = cleaned_text
            metadata = extract_metadata(cleaned_text)
            post_with_metadata = {**post, **metadata}
            enriched_posts.append(post_with_metadata)
        except Exception as e:
            print(f"Error processing post: {post['text'][:30]}... \nReason: {e}")

    # unify tag variants (e.g. "AI", "Ai", "ai" → "Artificial Intelligence")
    unified_tags = get_unified_tags(enriched_posts)
    for post in enriched_posts:
        tags = post.get("tags", [])
        post["tags"] = [unified_tags.get(tag, tag) for tag in tags]

    # Save output
    Path(processed_file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(processed_file_path, "w", encoding="utf-8") as f:
        json.dump(enriched_posts, f, indent=4, ensure_ascii=False)

# === 7. Run ===
if __name__ == "__main__":
    process_posts("data/raw_posts.json", "data/processed_posts.json")
