import os
import json
import pickle
import re
import ast

import gradio as gr
import numpy as np
import pandas as pd
from openai import OpenAI
from transformers import pipeline

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
CV_MODEL       = "juliaper/vit-food-category"

# ── Load ML model ─────────────────────────────────────────────────────────────
print("Loading ML difficulty model...")
with open("recipe_difficulty_model.pkl", "rb") as f:
    ml_artifact = pickle.load(f)
ml_model    = ml_artifact["model"]
ML_FEATURES = ml_artifact["features"]

# ── Load nutrition lookup (from second data source) ───────────────────────────
print("Loading nutrition data...")
with open("nutrition_lookup.pkl", "rb") as f:
    nutrition_lookup = pickle.load(f)

# ── Load recipe database (Data Source 1: Epicurious) ─────────────────────────
print("Loading recipe database...")
df_recipes = pd.read_csv("recipes_processed.csv")

# ── Load CV model ─────────────────────────────────────────────────────────────
print("Loading CV model...")
cv_classifier = pipeline("image-classification", model=CV_MODEL)

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

CATEGORIES = ["Chicken", "Bread", "Meat", "Soup", "Salad", "Dessert", "Seafood", "Pasta"]


# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(title, ingredients_str, instructions_str, category):
    try:
        ingredient_count = len(ast.literal_eval(str(ingredients_str)))
    except:
        ingredient_count = len(str(ingredients_str).split(','))

    instruction_words = len(str(instructions_str).split())
    instruction_steps = len(re.split(r'[.!?]+', str(instructions_str)))

    techniques = ['fold','whisk','knead','simmer','marinate','blanch','caramelize','deglaze']
    tech_features = {f'has_{t}': int(t in str(instructions_str).lower()) for t in techniques}

    cats = ['cake','cookie','bread','pasta','salad','soup','chicken','fish','beef','pizza','breakfast','steak']
    cat_features = {f'is_{c}': int(c in str(title).lower()) for c in cats}

    # Nutrition features from second data source
    nutr = nutrition_lookup.get(category, list(nutrition_lookup.values())[0])
    nutrition_features = {
        'nutr_caloric_value': nutr.get('Caloric Value', 0),
        'nutr_protein': nutr.get('Protein', 0),
        'nutr_fat': nutr.get('Fat', 0),
        'nutr_carbohydrates': nutr.get('Carbohydrates', 0),
        'nutr_dietary_fiber': nutr.get('Dietary Fiber', 0),
    }

    features = {
        'ingredient_count': ingredient_count,
        'instruction_words': instruction_words,
        'instruction_steps': instruction_steps,
        **tech_features,
        **cat_features,
        **nutrition_features
    }

    return pd.DataFrame([features])[ML_FEATURES]


# ── Find matching recipe ──────────────────────────────────────────────────────
def find_matching_recipe(category):
    keyword_map = {
        'chicken': 'chicken', 'bread': 'bread', 'meat': 'beef',
        'soup': 'soup', 'salad': 'salad', 'dessert': 'cake',
        'seafood': 'fish', 'pasta': 'pasta'
    }
    keyword = keyword_map.get(category.lower(), category.lower())
    matches = df_recipes[df_recipes['Title'].str.lower().str.contains(keyword, na=False)]
    if len(matches) == 0:
        matches = df_recipes
    return matches.sample(1, random_state=42).iloc[0]


# ── LLM explanation (Prompt 1) ────────────────────────────────────────────────
def generate_explanation_v1(category, difficulty, recipe_title, ingredients, instructions, nutrition):
    if not openai_client:
        return "OpenAI API key not set."

    system_prompt = """You are a friendly cooking assistant.
Given food category, difficulty, recipe details and nutrition info, write a warm 3-4 sentence explanation.
Mention the dish, difficulty level, 1-2 cooking tips, and the nutritional highlights.
Be encouraging and fun!"""

    user_prompt = f"""Category: {category}
Difficulty: {difficulty}
Recipe: {recipe_title}
Nutrition: {nutrition['Caloric Value']:.0f} kcal, {nutrition['Protein']:.1f}g protein, {nutrition['Fat']:.1f}g fat
Instructions preview: {str(instructions)[:300]}

Write an engaging explanation with cooking tip and nutrition highlight."""

    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content


# ── LLM explanation (Prompt 2 — different style for comparison) ───────────────
def generate_explanation_v2(category, difficulty, recipe_title, ingredients, instructions, nutrition):
    if not openai_client:
        return "OpenAI API key not set."

    system_prompt = """You are a professional chef and nutritionist.
Provide a structured analysis of the recipe in bullet points covering:
- Difficulty assessment
- Key technique required
- Nutritional value
- Health tip
Be concise and professional."""

    user_prompt = f"""Analyze this recipe:
Category: {category} | Difficulty: {difficulty}
Recipe: {recipe_title}
Nutrition per serving: {nutrition['Caloric Value']:.0f} kcal, {nutrition['Protein']:.1f}g protein

Provide structured bullet point analysis."""

    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content


# ── Main pipeline ─────────────────────────────────────────────────────────────
def analyze_food_image(image, prompt_style):
    if image is None:
        return "Please upload an image.", {}, "Easy", {}, "", "", ""

    try:
        # Step 1: CV — classify food category (Data Source 1 images)
        cv_results = cv_classifier(image)
        category   = cv_results[0]['label']
        cv_scores  = {r['label']: round(r['score'], 3) for r in cv_results}

        # Step 2: Find matching recipe (Data Source 1: Epicurious)
        recipe = find_matching_recipe(category)

        # Step 3: Get nutrition data (Data Source 2: Food Nutrition Dataset)
        nutrition = nutrition_lookup.get(category, list(nutrition_lookup.values())[0])
        nutrition_text = (
            f"**{category} — Nutrition per 100g (category average):**\n"
            f"- Calories: {nutrition['Caloric Value']:.0f} kcal\n"
            f"- Protein: {nutrition['Protein']:.1f}g\n"
            f"- Fat: {nutrition['Fat']:.1f}g\n"
            f"- Carbs: {nutrition['Carbohydrates']:.1f}g\n"
            f"- Fiber: {nutrition['Dietary Fiber']:.1f}g\n"
            f"\nNote: Values are averages for the {category} category."
        )

        # Step 4: ML — predict difficulty using both data sources
        features   = extract_features(
            recipe['Title'], recipe['Ingredients'],
            recipe['Instructions'], category
        )
        difficulty = ml_model.predict(features)[0]
        diff_proba = ml_model.predict_proba(features)[0]
        diff_scores = {cls: round(prob, 3) for cls, prob in zip(ml_model.classes_, diff_proba)}

        # Step 5: NLP — generate explanation (compare 2 prompts)
        if prompt_style == "Friendly (Prompt 1)":
            explanation = generate_explanation_v1(
                category, difficulty, recipe['Title'],
                recipe['Ingredients'], recipe['Instructions'], nutrition
            )
        else:
            explanation = generate_explanation_v2(
                category, difficulty, recipe['Title'],
                recipe['Ingredients'], recipe['Instructions'], nutrition
            )

        recipe_text = f"**{recipe['Title']}**\n\n{str(recipe['Instructions'])[:600]}..."

        return (
            f"**{category}** (confidence: {cv_results[0]['score']:.1%})",
            cv_scores,
            f"**{difficulty}**",
            diff_scores,
            nutrition_text,
            explanation,
            recipe_text
        )

    except Exception as e:
        return f"Error: {str(e)}", {}, "Unknown", {}, "", "", ""


# ── Gradio Interface ──────────────────────────────────────────────────────────
with gr.Blocks(title="AI Recipe Assistant") as demo:
    gr.Markdown(
        """
        # AI Recipe Assistant
        Upload a food photo to get:
        - **Food Category** — Computer Vision (ViT fine-tuned on Epicurious images)
        - **Recipe Difficulty** — ML model (trained on Epicurious + Nutrition data)
        - **Nutrition Info** — from Food Nutrition Dataset (2,395 foods)
        - **Recipe & Tips** — LLM explanation (compare 2 prompt styles!)
        """
    )

    with gr.Row():
        with gr.Column():
            image_input  = gr.Image(type="filepath", label="Upload Food Photo")
            prompt_style = gr.Radio(
                choices=["Friendly (Prompt 1)", "Professional (Prompt 2)"],
                value="Friendly (Prompt 1)",
                label="LLM Prompt Style"
            )
            analyze_btn = gr.Button("Analyze", variant="primary")

            gr.Examples(
                examples=[
                    ["example_images/dessert.jpg", "Friendly (Prompt 1)"],
                    ["example_images/pasta.jpg", "Professional (Prompt 2)"],
                    ["example_images/salad.jpg", "Friendly (Prompt 1)"],
                    ["example_images/soup.jpg", "Professional (Prompt 2)"],
                    ["example_images/chicken.jpg", "Friendly (Prompt 1)"],
                ],
                inputs=[image_input, prompt_style],
            )

        with gr.Column():
            category_output   = gr.Textbox(label="Food Category (CV Model)")
            cv_scores_output  = gr.Label(label="Category Confidence Scores")
            difficulty_output = gr.Textbox(label="Recipe Difficulty (ML Model)")
            diff_scores_output = gr.Label(label="Difficulty Scores")
            nutrition_output  = gr.Textbox(label="Nutrition Info (Food Nutrition Dataset)", lines=7)

    explanation_output = gr.Textbox(label="Recipe Tips & Explanation (LLM)", lines=5)
    recipe_output      = gr.Textbox(label="Suggested Recipe", lines=5)

    analyze_btn.click(
        fn=analyze_food_image,
        inputs=[image_input, prompt_style],
        outputs=[
            category_output, cv_scores_output,
            difficulty_output, diff_scores_output,
            nutrition_output, explanation_output, recipe_output
        ]
    )

    gr.Markdown(
        """
        ---
        **Data Source 1:** Food Ingredients and Recipe Dataset with Images — Epicurious (13,501 recipes)
        **Data Source 2:** Food Nutrition Dataset (2,395 foods with 35 nutritional features)
        **CV Model:** ViT fine-tuned on 8 food categories (1,600 training images)
        **ML Model:** Gradient Boosting Classifier (26 features from both data sources)
        **NLP:** OpenAI gpt-4.1-mini with 2 different prompt styles for comparison
        """
    )

demo.launch()
