# 🔍 Semantic Screenshot Engine

A smart, AI-powered search engine that allows you to find images based on their **content and meaning**, not just their filenames. Built with Python, Streamlit, and OpenAI's CLIP model.

### 🚀 **Live Demo:** [Click here to try the App](https://semantic-screenshot-engine-ritipw.streamlit.app/)

---

## 🧐 The Problem
We all have thousands of screenshots, memes, and photos on our devices with useless names like `IMG_20251230.jpg` or `Screenshot 54.png`. Finding a specific image ("that error message from last week" or "the recipe I saved") is impossible using standard keyword search.

## 💡 The Solution
This project uses **Multimodal AI (CLIP)** to "see" and "understand" the contents of your images. It maps both images and text into a shared vector space, allowing you to search for concepts.

**Example:**
* You upload a photo of a cat sleeping on a sofa.
* You search for: *"lazy pet"*
* **Result:** The AI finds the image, even though the word "lazy" or "pet" isn't written anywhere in the file.

---

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/) (for the Web UI & Mobile compatibility)
* **AI Model:** [CLIP (Contrastive Language-Image Pre-Training)](https://github.com/openai/CLIP) via `sentence-transformers`
* **Backend Logic:** Python, PyTorch, NumPy
* **Image Processing:** Pillow (PIL)

---

## 📸 Features

* **Semantic Search:** Search by description (e.g., "coding error", "beach sunset", "financial document").
* **Privacy-First Design:** Images are processed in-memory during the session and are not permanently stored on the server.
* **Mobile Friendly:** Works on mobile browsers for searching your camera roll.
* **Drag & Drop:** Easy interface to upload multiple images at once.

---

## 💻 How to Run Locally

If you want to run this project on your own machine (and search local folders without uploading), follow these steps:

**1. Clone the repository**
```bash
git clone [https://github.com/your-username/semantic-screenshot-engine.git](https://github.com/your-username/semantic-screenshot-engine.git)
cd semantic-screenshot-engine