#!/usr/bin/env python3
"""Convert [Author, year] citations in manuscript/draft.md to Elsevier numbered
style (order of first appearance) and append the reference list.
Output: manuscript/draft_numbered.md"""
import re, sys
src = open("manuscript/draft.md").read()
pool_txt = open("manuscript/references_pool.md").read()
pool = {}
for m in re.finditer(r'^\[(\d+)\]\s+(.*?)(?=\n\[\d+\]|\Z)', pool_txt, re.S | re.M):
    pool[int(m.group(1))] = " ".join(m.group(2).split())

NEW = {
 "DeYoung et al., 2020": 'J. DeYoung, S. Jain, N. F. Rajani, E. Lehman, C. Xiong, R. Socher, and B. C. Wallace, "ERASER: A benchmark to evaluate rationalized NLP models," in Proc. ACL, 2020, pp. 4443–4458.',
 "Guo et al., 2017": 'C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in Proc. ICML, 2017, pp. 1321–1330.',
 "Holm, 1979": 'S. Holm, "A simple sequentially rejective multiple test procedure," Scandinavian Journal of Statistics, vol. 6, no. 2, pp. 65–70, 1979.',
 "Moustafa and Slay, 2015": 'N. Moustafa and J. Slay, "UNSW-NB15: A comprehensive data set for network intrusion detection systems," in Proc. Military Communications and Information Systems Conference (MilCIS), IEEE, 2015, pp. 1–6.',
 "Petsiuk et al., 2018": 'V. Petsiuk, A. Das, and K. Saenko, "RISE: Randomized input sampling for explanation of black-box models," in Proc. BMVC, 2018.',
 "Racha et al., 2024": '[VERIFY] S. Racha et al., "Deep-IDS: A deep learning-based intrusion detection system," 2024.',
 "Samek et al., 2017": 'W. Samek, A. Binder, G. Montavon, S. Lapuschkin, and K.-R. Müller, "Evaluating the visualization of what a deep neural network has learned," IEEE Transactions on Neural Networks and Learning Systems, vol. 28, no. 11, pp. 2660–2673, 2017.',
 "Tuli et al., 2022": 'S. Tuli, G. Casale, and N. R. Jennings, "TranAD: Deep Transformer networks for anomaly detection in multivariate time series data," Proceedings of the VLDB Endowment, vol. 15, no. 6, pp. 1201–1214, 2022.',
}
MAP = {"Abnar and Zuidema, 2020":51,"Alanazi et al., 2025":65,"Arik and Pfister, 2021":25,"Boukela et al., 2024":27,
 "Capuano et al., 2022":6,"Chen et al., 2024":57,"Ferrag et al., 2020":4,"Gorishniy et al., 2021":10,"Grinsztajn et al., 2022":49,
 "Hooker et al., 2019":45,"Huang et al., 2021":11,"Imrana et al., 2022":22,"Lin et al., 2017":34,"Lin et al., 2023":40,
 "Liu and Wu, 2023":35,"Lundberg and Lee, 2017":14,"Mane and Rao, 2021":43,"McElfresh et al., 2023":59,"Mishra et al., 2019":42,
 "Pawlicki et al., 2024":7,"Ribeiro et al., 2016":15,"Roponena et al., 2024":44,"Sharafaldin et al., 2018":16,
 "Somepalli et al., 2021":26,"Tavallaee et al., 2009":38,"Thakkar and Lohiya, 2020":71,"Vinayakumar et al., 2019":2,
 "Wang et al., 2023":56,"Zebin et al., 2022":66,"Zhang et al., 2025":62,"Yang et al., 2016":24,"Jain and Wallace, 2019":47,
 "Wiegreffe and Pinter, 2019":48,"Chefer et al., 2021":60,"Vaswani et al., 2017":9,"Akiba et al., 2019":52,
 "Loshchilov and Hutter, 2019":53,"Loshchilov and Hutter, 2017":50,"Chawla et al., 2002":29,"Hendrycks and Gimpel, 2016":55,
 "Devlin et al., 2019":54}

# narrative "Name et al. [2016]" / "Jain and Wallace [2019]" -> parenthetical key
def narr(m):
    return f"[{m.group(1)}, {m.group(2)}]"
txt = re.sub(r'\b([A-Z][A-Za-z\-]+(?: et al\.| and [A-Z][A-Za-z\-]+)?) \[((?:19|20)\d{2})\]', narr, src)

order, numbers = [], {}
def num(key):
    key = key.strip()
    if key not in numbers:
        if key not in MAP and key not in NEW:
            sys.exit(f"UNMAPPED citation key: {key!r}")
        order.append(key); numbers[key] = len(order)
    return numbers[key]

def repl(m):
    keys = [k.strip() for k in m.group(1).split(';')]
    ns = sorted(num(k) for k in keys)
    return "[" + ",".join(str(n) for n in ns) + "]"
txt = re.sub(r'\[((?:[A-Z][^\[\]]*?(?:19|20)\d{2}[a-z]?)(?:;\s*[^\[\]]*?(?:19|20)\d{2}[a-z]?)*)\]', repl, txt)

refs = ["## References", ""]
for i, key in enumerate(order, 1):
    entry = NEW[key] if key in NEW else pool[MAP[key]]
    refs.append(f"[{i}] {entry}")
    refs.append("")
txt = txt[:txt.find("## References")] + "\n".join(refs) + "\n"
open("manuscript/draft_numbered.md", "w").write(txt)
left = re.findall(r'\[[^\]\d][^\]]*(?:19|20)\d{2}[^\]]*\]', txt)
print(f"{len(order)} references; unconverted author-year brackets: {left[:5]}")
