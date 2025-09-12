import gradio as gr
from pathlib import Path
from packaging import version

from src.hmm_decoder import HMMSpellChecker
from src.diff import mark_diff

# ---------- Model ----------
ARTIFACT_DIR = Path("artifacts")
checker = HMMSpellChecker.from_artifacts(str(ARTIFACT_DIR))

# ---------- Helpers ----------
def count_changes(a: str, b: str) -> int:
    ta, tb = a.split(), b.split()
    n = min(len(ta), len(tb))
    return sum(1 for i in range(n) if ta[i] != tb[i]) + abs(len(ta) - len(tb))

def correct_api(text: str):
    text = (text or "").strip()
    if not text:
        return "", "", "", gr.update(value="0"), gr.update(visible=False, value="")
    fixed = checker.correct(text)
    orig_mark, corr_mark = mark_diff(text, fixed)
    diffs = count_changes(text, fixed)
    status_md = f"<div><strong>Số lỗi: {diffs}</strong></div>"
    return fixed, orig_mark, corr_mark, str(diffs), gr.update(visible=True, value=status_md)

EXAMPLES = [
    ["Toi rat thich doc sach va di dao buoi toi."],
    ["Mot so tu tieng Viet co dau rat de bi go sai vi telex."],
    ["Thong qua cong tac tuyen truyeenf, van dong…"],
]

# ---------- UI ----------
DARK_CSS = """
:root { --radius-xl: 14px; }
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; }
body, .gradio-container { background:#0f1220; color:#e8eaf2; }
.header h1 { margin:0; font-weight:800; letter-spacing:.2px; }
.badge { font-size:12px; padding:4px 10px; border-radius:999px; background:#2a2f45; color:#cfd3ff; }

.card,
textarea, .prose, .markdown, .terminal {
  background:#1b2133 !important; color:#e8eaf2 !important;
  border:1px solid #2a2f45 !important; border-radius:14px !important;
}
.prose p, .prose li, .prose strong, .prose em, .markdown { color:#e8eaf2 !important; }
.prose code, pre code { background:#111525 !important; color:#e8eaf2 !important; }

label, .block .label-wrap span { color:#cfd3ff !important; font-weight:700; }
input, textarea { caret-color:#cfd3ff; }
textarea::placeholder { color:#98a0c6 !important; }

button.svelte-1ipelgc, .gr-button { border-radius:12px !important; }
.copy-btn, .icon-btn { filter: invert(1) hue-rotate(180deg); }
.footer { color:#98a0c6; font-size:12px; text-align:center; margin-top:12px; }
#right-col .wrap { gap: 10px !important; }

#status-panel .prose, #status-panel .prose p {
  text-align: center !important;
  font-weight: 700 !important;
  color: #cfd3ff !important;
}
"""

with gr.Blocks(css=DARK_CSS, fill_height=True) as demo:
    gr.HTML(
        """
        <div class="header" style="display:flex;align-items:center;gap:12px;">
          <h1>ViSpell — Vietnamese Spelling Correction (HMM Trigram)</h1>
          <span class="badge">FastAPI/Gradio • Viterbi decoding</span>
        </div>
        <p style="margin-top:6px;color:#cfd3ff">
          Nhập văn bản tiếng Việt. Hệ thống sẽ sửa lỗi chính tả và <b>highlight</b> khác biệt.
        </p>
        """
    )

    with gr.Row(equal_height=True):
        # -------- Left
        with gr.Column(scale=5, min_width=420):
            inp = gr.Textbox(
                label="Nhập câu/đoạn văn",
                lines=8,
                placeholder="Dán văn bản của bạn vào đây…",
                elem_classes=["card"],
            )

            with gr.Row():
                btn_correct = gr.Button("🔧 Sửa chính tả", variant="primary")
                btn_clear = gr.Button("🧹 Xoá", variant="secondary")

            with gr.Accordion("Ví dụ nhanh", open=False):
                gr.Examples(examples=EXAMPLES, inputs=[inp], examples_per_page=6)

            with gr.Row():
                changes = gr.Textbox(
                    label="Số vị trí thay đổi", value="0", interactive=False, elem_classes=["card"]
                )
                status_panel = gr.Markdown("", visible=False, elem_classes=["card"], elem_id="status-panel")

        # -------- Right
        with gr.Column(scale=7, min_width=520, elem_id="right-col"):
            out_fixed = gr.Textbox(
                label="Kết quả (copy-friendly)",
                lines=8,
                show_copy_button=True,
                elem_classes=["card"],
                interactive=True,            # <-- cho phép chỉnh sửa ô kết quả
            )
            with gr.Row():
                out_orig_mark = gr.Markdown(label="Khác biệt (gốc)", elem_classes=["card"])
                out_corr_mark = gr.Markdown(label="Khác biệt (đã sửa)", elem_classes=["card"])

            # nút đổ ngược kết quả sang input (tuỳ chọn)
            use_btn = gr.Button("↩ Dùng kết quả làm đầu vào")

    gr.HTML('<div class="footer">© 2025 ViSpell HMM — Demo for recruiters • Made with ❤️</div>')

    # wiring
    btn_correct.click(
        fn=correct_api,
        inputs=[inp],
        outputs=[out_fixed, out_orig_mark, out_corr_mark, changes, status_panel],
        queue=True,
    )
    btn_clear.click(
        fn=lambda: ("", "", "", "0", gr.update(value="", visible=False)),
        inputs=None,
        outputs=[out_fixed, out_orig_mark, out_corr_mark, changes, status_panel],
    )
    # chuyển giá trị đã chỉnh ở ô kết quả sang input
    use_btn.click(fn=lambda t: t, inputs=[out_fixed], outputs=[inp])

if __name__ == "__main__":
    if version.parse(gr.__version__).major >= 4:
        demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)
    else:
        demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)
