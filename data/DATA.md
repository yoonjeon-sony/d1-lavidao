### ThinkMorph
- subsets: Chart-Refocus, Jigsaw Assembly, Spatial Navigation, Visual Search
- keys: pid, question, answer, problem_image_0, reasoning_image_0, bboxDiff (except for Jigsaw Assembly), full_text_only_thought


### Zebra-CoT
- subsets:
    - 2D Visual Reasoning
    - 3D Visual Reasoning
    - Scientific Reasoning
    - Visual Logic Strategic Games
- keys: id, question, answer, problem_image, reasoning_image, full_text_only_thought

### Math-VR-Train (Excluded / Process later)
- subsets: null
- keys: id, question, category (only "multimodal" contains problem image), analysis, image1 (problem image), imageX (reasoning image), bboxDiff_XX, **Therefore, the answers are:**\n\n- 
- Preprocessing
    - analysis: Need processing 
    - (remove ### Problem Analysis\n\n,  **Question (XX):**, ### Solution Explanation\n\n, \n\n<imageXX>\n\n, )
    ``` 
    "### Problem Analysis\n\nAccording to the definition of fractions, if the unit \"1\" is evenly divided into several equal parts, the number representing one or more of these parts is called a **fraction**.\n\n1. **Question (1):**\n    - Take the whole large triangle as unit \"1\"; it is evenly divided into 4 parts, and the shaded part occupies 1 of these parts.\n    - Written as a fraction: $\\frac{1}{4}$\n\n2. **Question (2):**\n    - Take the circle as unit \"1\"; it is evenly divided into 2 parts, with 1 part shaded.\n    - Written as a fraction: $\\frac{1}{2}$\n\n3. **Question (3):**\n    - Take the rectangle as unit \"1\"; it is evenly divided into 8 parts, with 4 shaded parts.\n    - Written as a fraction: $\\frac{4}{8}$\n\n4. **Question (4):**\n    - Take the whole hexagon as unit \"1\"; it is evenly divided into 6 parts, and the shaded part occupies 2 of these parts.\n    - Written as a fraction: $\\frac{2}{6}$\n\n5. **Question (5):**\n    - Take the rectangle as unit \"1\"; it is evenly divided into 3 parts, with 1 part shaded.\n    - Written as a fraction: $\\frac{1}{3}$\n\n6. **Question (6):**\n    - Take the entire figure as unit \"1\"; it is evenly divided into 13 parts, and the shaded part occupies 5 of these parts.\n    - Written as a fraction: $\\frac{5}{13}$\n\n7. **Question (7):**\n    - Take the entire figure as unit \"1\"; it is evenly divided into 10 parts, and the shaded part occupies 6 of these parts.\n    - Written as a fraction: $\\frac{6}{10}$\n\n### Solution Explanation\n\nThe shaded portions in the figures can be represented by the following fractions:\n\n<image2>\n\n**Therefore, the answers are:**\n\n- $\\frac{1}{4}$\n- $\\frac{1}{2}$\n- $\\frac{4}{8}$\n- $\\frac{2}{6}$\n- $\\frac{1}{3}$\n- $\\frac{5}{13}$\n- $\\frac{6}{10}$\n
    ```
