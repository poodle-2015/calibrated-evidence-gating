# Evidence Gate prompt

| File | Method | Output format |
| --- | --- | --- |
| `gate_system.txt` + `gate_user_template.txt` | Evidence Gate (fine-tuned classifier) | strict JSON: `sufficiency` + `diagnosis_type` |

Placeholders `{context}` and `{question}` are filled by `cegate/prompts.py`. Context lines use
the paper's `[Title#sent_id] sentence` format (with `[TITLE] Title` headers), preserving the
original sentence number for every support pointer.

Gate scoring does not parse generated text. Given the common prefix
`{"sufficiency": "`, `cegate/infer_gate.py` compares the **average** log-likelihood of the two
label words `sufficient` and `insufficient` (score = avg logP(insufficient) − avg logP(sufficient)).
