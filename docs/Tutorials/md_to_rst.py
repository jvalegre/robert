import os
import re

def clean_text(text):
    text = text.replace("<br><br>", "\n\n")
    text = text.replace("<br>", "\n")

    # bold
    text = re.sub(
        r"<\s*b\s*>(.*?)<\s*/\s*b\s*>",
        r"**\1**",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    # italic
    text = re.sub(
        r"<\s*i\s*>(.*?)<\s*/\s*i\s*>",
        r"*\1*",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    return text.strip()

def sort_key(filename):
    name = filename.replace(".png", "")
    numbers = re.findall(r'\d+', name)
    return tuple(int(n) for n in numbers)

def indent_block(text, spaces=3):
    prefix = " " * spaces
    return "\n".join(prefix + line if line.strip() else "" for line in text.split("\n"))

def convert(md_path, img_folder, out_path, prefix):
    print(f"\n===== {prefix.upper()} =====")

    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    steps = [clean_text(s) for s in content.split('---') if s.strip()]

    images = sorted(
        [f for f in os.listdir(img_folder) if f.endswith(".png")],
        key=sort_key
    )

    print(f"Steps: {len(steps)}")
    print(f"Images: {len(images)}")

    rst = ""

    # ONE container per tutorial
    rst += f".. container:: step\n\n"

    for i, step in enumerate(steps):
        step_id = i + 1

        # STEP WRAPPER
        rst += "   .. raw:: html\n\n"
        rst += f"      <div class='step-content' data-prefix='{prefix}' data-step='{step_id}'>\n\n"

        # IMAGE (BIG, CLEAN)
        if i < len(images):
            rst += f"   .. figure:: tutorial_images/{prefix}/{images[i]}\n\n"
        else:
            print(f"❌ Missing image → Step {step_id}")

        # TEXT
        rst += indent_block(step, 3) + "\n\n"

        # NAV BUTTONS
        rst += "   .. raw:: html\n\n"

        buttons = '      <div class="step-nav">\n'

        if step_id > 1:
            buttons += f'         <button onclick="prevStep(\'{prefix}\',{step_id})">Previous</button>\n'

        if step_id < len(steps):
            buttons += f'         <button onclick="nextStep(\'{prefix}\',{step_id})">Next</button>\n'

        buttons += "      </div>\n"

        rst += buttons + "\n\n"

        # CLOSE STEP
        rst += "   .. raw:: html\n\n"
        rst += "      </div>\n\n"

    if len(images) > len(steps):
        print(f"⚠️ Extra images: {len(images) - len(steps)}")

    print(f"Done: {prefix}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(rst)


# RUN ALL
convert("chemdraw.md", "tutorial_images/chemdraw", "chemdraw.rst", "chemdraw")
convert("csv.md", "tutorial_images/csv", "csv.rst", "csv")
convert("descriptors.md", "tutorial_images/descriptors", "descriptors.rst", "descriptors")
convert("overview.md", "tutorial_images/overview", "overview.rst", "overview")
convert("predictions.md", "tutorial_images/predictions", "predictions.rst", "predictions")