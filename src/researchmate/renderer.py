
import re
import streamlit as st


# Reconnaître les différents délimiteurs LaTeX.
MATH_PATTERN = re.compile(
    r"\$\$(.*?)\$\$"
    r"|\\\[(.*?)\\\]"
    r"|\\\((.*?)\\\)"
    r"|(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)",
    re.DOTALL,
)


def render_answer(text: str) -> None:
    """Affiche le texte Markdown et compile les formules LaTeX."""

    parts = MATH_PATTERN.split(text)

    # Les groupes capturés suivent cet ordre :
    # $$...$$, \[...\], \(...\), $...$
    for index in range(0, len(parts), 5):
        prose = parts[index]

        if prose.strip():
            st.markdown(prose)

        if index + 1 < len(parts) and parts[index + 1] is not None:
            formula = parts[index + 1].strip()
            if formula:
                st.latex(formula)

        if index + 2 < len(parts) and parts[index + 2] is not None:
            formula = parts[index + 2].strip()
            if formula:
                st.latex(formula)

        if index + 3 < len(parts) and parts[index + 3] is not None:
            formula = parts[index + 3].strip()
            if formula:
                st.latex(formula)

        if index + 4 < len(parts) and parts[index + 4] is not None:
            formula = parts[index + 4].strip()
            if formula:
                st.latex(formula)
