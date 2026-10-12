
import os
import json
import sqlite3
from datetime import date

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader
from docx import Document

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
client = genai.Client(api_key=API_KEY) if API_KEY else None

DB = "attendance.db"
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}


# 1. Create the attendance database
def init_db():
    with sqlite3.connect(DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student TEXT NOT NULL,
                subject TEXT NOT NULL,
                day TEXT NOT NULL,
                status TEXT NOT NULL,
                UNIQUE(student, subject, day)
            )
        """)


init_db()


# 2. Read uploaded documents
def read_document(file):
    filename = file.filename.lower()

    if filename.endswith(".txt"):
        return file.read().decode("utf-8", errors="replace")

    if filename.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    if filename.endswith(".docx"):
        document = Document(file)
        return "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

    raise ValueError("Only PDF, DOCX, and TXT are supported.")


# 3. Calculator Tool
def calculator_tool(attended, total):
    if total <= 0 or attended < 0 or attended > total:
        return "Please provide valid attendance counts."

    percentage = attended / total * 100
    return f"Attendance percentage: {percentage:.2f}%"


# 4. Attendance Tool
def attendance_tool(student, subject, day, status):
    if not student or not subject:
        return "Please provide the student name and subject."

    if status not in ["Present", "Absent"]:
        return "Status must be Present or Absent."

    try:
        date.fromisoformat(day)
    except (TypeError, ValueError):
        return "Use the date format YYYY-MM-DD."

    try:
        with sqlite3.connect(DB) as conn:
            conn.execute("""
                INSERT INTO attendance
                (student, subject, day, status)
                VALUES (?, ?, ?, ?)
            """, (student, subject, day, status))

        return f"Attendance marked: {student}, {subject}, {day}, {status}."

    except sqlite3.IntegrityError:
        return "Attendance is already recorded for that date."


# 5. Ask Gemini to choose a tool
def choose_tool(question, document):
    prompt = f"""
You are the routing assistant for a college AI system.

Choose one route:
calculator, attendance_mark, timetable, document_qa, general.

Rules:
- Attendance percentage from attended and total counts:
  calculator.
- Mark attendance for a specific day:
  attendance_mark.
- Timetable questions:
  timetable.
- Questions about uploaded documents:
  document_qa.
- Other questions:
  general.

Return only JSON with these fields:
route, attended, total, student, subject, day, status.

Use null for missing information.
Use YYYY-MM-DD for dates.
Status must be Present or Absent.

Today's date is {date.today().isoformat()}.
Question: {question}
Uploaded document: {document[:12000]}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0
        )
    )

    return json.loads(response.text)


# 6. Generate an answer using Gemini
def ask_gemini(question, document):
    response = client.models.generate_content(
        model=MODEL,
        contents=f"""
You are a helpful College AI Assistant.
Answer clearly and accurately.
Use the document when relevant.
Do not invent timetable or college information.

Document:
{document[:12000]}

Student question:
{question}
"""
    )
    return response.text or "No answer was generated."


# 7. Main website page
@app.route("/")
def home():
    return render_template("index.html")


# 8. Process student questions
@app.route("/api/chat", methods=["POST"])
def chat():
    question = request.form.get("question", "").strip()
    student = request.form.get("student", "").strip()
    subject = request.form.get("subject", "").strip()
    day = request.form.get("day", "").strip() or date.today().isoformat()
    status = request.form.get("status", "Present")

    uploaded_file = request.files.get("file")
    document = ""
    filename = ""

    if uploaded_file and uploaded_file.filename:
        filename = uploaded_file.filename
        if "." not in filename or filename.rsplit(".", 1)[1].lower() not in ALLOWED_EXTENSIONS:
            return jsonify({
                "answer": "Upload a PDF, DOCX, or TXT file.",
                "tool": "File Validation"
            }), 400

        try:
            document = read_document(uploaded_file)
        except Exception as error:
            return jsonify({
                "answer": f"Could not read file: {error}",
                "tool": "Document Reader"
            }), 400

    if not question and not document:
        return jsonify({
            "answer": "Enter a question or upload a file.",
            "tool": "Input Validation"
        }), 400

    if not client:
        return jsonify({
            "answer": "Add your Gemini API key to the .env file.",
            "tool": "Configuration"
        }), 500

    try:
        route = choose_tool(question, document)
        selected = route.get("route", "general")

        if selected == "calculator":
            attended = route.get("attended")
            total = route.get("total")

            if attended is None or total is None:
                answer = (
                    "Please tell me the number of classes attended "
                    "and the total classes conducted."
                )
            else:
                answer = calculator_tool(int(attended), int(total))

            tool = "Calculator Tool"

        elif selected == "attendance_mark":
            name = route.get("student") or student
            subj = route.get("subject") or subject
            attendance_day = route.get("day") or day
            attendance_status = route.get("status") or status

            answer = attendance_tool(
                name, subj, attendance_day, attendance_status
            )
            tool = "Attendance Tool"

        elif selected == "timetable":
            answer = ask_gemini(question, document)
            tool = "Timetable Analysis Tool + Gemini"

        elif selected == "document_qa":
            answer = ask_gemini(
                question or "Summarize this document.", document
            )
            tool = "Document Reader + Gemini"

        else:
            answer = ask_gemini(question, document)
            tool = "Gemini"

        if filename:
            answer += f"\n\nFile processed: {filename}"

        return jsonify({"answer": answer, "tool": tool})

    except Exception as error:
        return jsonify({
            "answer": f"Request failed: {error}",
            "tool": "Gemini API"
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
