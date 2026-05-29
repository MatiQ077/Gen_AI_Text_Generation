"""
app.py — Gradio web UI for the RAG-Enhanced Travel Itinerary Generator.

Entry point for the application.  Builds a two-column Gradio Blocks interface:
    Left column:  seven input fields (destination, trip length, budget,
                  interests, travel style, pace, special preferences).
    Right column: Markdown output panel displaying the generated itinerary.

All generation logic is delegated to src/rag.py (generate_itinerary), which
handles FAISS retrieval, prompt construction, and GPT-2 inference.

Usage:
    python app.py

Prerequisites:
    1. Corpora processed:  python src/clean_data.py --mode city (and itineraries, pdf)
    2. FAISS index built:  python src/build_index.py
    3. GPT-2 fine-tuned:   python src_gpt/train.py
"""

import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from rag import generate_itinerary

# Validate the inpits and call generate_itinerary
def run(destination, trip_length, budget, interests, travel_style, pace, special_prefs):
    
    if not destination.strip():
        return "Please enter a destination."
    return generate_itinerary(
        destination, trip_length, budget, interests,
        travel_style, pace, special_prefs,
    )

with gr.Blocks(title="Travel Itinerary Generator") as demo:
    gr.Markdown("# RAG-Enhanced Travel Itinerary Generator")
    gr.Markdown("Enter your trip details to get a personalized day-by-day itinerary.")

    with gr.Row():
        with gr.Column():
            destination = gr.Textbox(
                label="Destination",
                placeholder="e.g. Tokyo",
            )
            trip_length = gr.Dropdown(
                ["3 days", "5 days", "7 days", "10 days"],
                value="5 days",
                label="Trip Length",
            )
            budget = gr.Dropdown(
                ["budget", "moderate", "luxury"],
                value="moderate",
                label="Budget",
            )
            interests = gr.Textbox(
                label="Interests",
                placeholder="e.g. temples, food, hiking",
            )
            travel_style = gr.Dropdown(
                ["cultural", "adventure", "relaxed", "foodie"],
                value="cultural",
                label="Travel Style",
            )
            pace = gr.Dropdown(
                ["relaxed", "moderate", "packed"],
                value="moderate",
                label="Pace",
            )
            special_prefs = gr.Textbox(
                label="Special Preferences (optional)",
                placeholder="e.g. vegetarian, no museums",
            )
            btn = gr.Button("Generate Itinerary", variant="primary")

        with gr.Column():
            output = gr.Markdown(label="Your Itinerary")

    btn.click(
        fn=run,
        inputs=[destination, trip_length, budget, interests,
                travel_style, pace, special_prefs],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch()