#!/usr/bin/env python3
"""[Author, year] -> numbered citations (order of first appearance) for manuscript/scirep_draft.md."""
import re, sys
src = open("manuscript/scirep_draft.md").read()
pool_txt = open("manuscript/references_pool.md").read()
pool = {int(m.group(1)): " ".join(m.group(2).split()) for m in re.finditer(r'^\[(\d+)\]\s+(.*?)(?=\n\[\d+\]|\Z)', pool_txt, re.S | re.M)}
# corrections to pool entries (verified Oct 2026)
pool[4] = pool[4].replace("C. L. Ferrag", "M. A. Ferrag")
pool[52] = pool[52].replace("M. Kido", "M. Koyama")
NEW = {
 "Arp et al., 2022": 'D. Arp, E. Quiring, F. Pendlebury, A. Warnecke, F. Pierazzi, C. Wressnegger, L. Cavallaro, and K. Rieck, "Dos and don\'ts of machine learning in computer security," in Proc. USENIX Security Symposium, 2022, pp. 3971–3988.',
 "Engelen et al., 2021": 'G. Engelen, V. Rimmer, and W. Joosen, "Troubleshooting an intrusion detection dataset: the CICIDS2017 case," in Proc. IEEE Security and Privacy Workshops (SPW), 2021, pp. 7–12.',
 "Warnecke et al., 2020": 'A. Warnecke, D. Arp, C. Wressnegger, and K. Rieck, "Evaluating explanation methods for deep learning in security," in Proc. IEEE European Symposium on Security and Privacy (EuroS&P), 2020, pp. 158–174.',
 "Arreche et al., 2024": 'O. Arreche, T. R. Guntur, J. W. Roberts, and M. Abdallah, "E-XAI: Evaluating black-box explainable AI frameworks for network intrusion detection," IEEE Access, vol. 12, pp. 23954–23988, 2024. DOI: 10.1109/ACCESS.2024.3365140.',
 "Racherla et al., 2024": '[VERIFY pages/DOI] S. Racherla, P. Sripathi, N. Faruqui, M. A. Kabir, M. Whaiduzzaman, and S. A. Shah, "Deep-IDS: A real-time intrusion detector for IoT nodes using deep learning," IEEE Access, vol. 12, 2024.',
 "NSM dataset, 2026": 'N. S. Mert, F. Şahin, and B. Yılmaz, "NSM-IDS-2026: A CICIDS2017-compatible multi-class intrusion detection dataset from an isolated laboratory," Zenodo, 2026. DOI: 10.5281/zenodo.23392183 (version 1.2.0 used: 10.5281/zenodo.23449599).',
 "Moustafa and Slay, 2015": 'N. Moustafa and J. Slay, "UNSW-NB15: A comprehensive data set for network intrusion detection systems," in Proc. Military Communications and Information Systems Conference (MilCIS), IEEE, 2015, pp. 1–6.',
 "Samek et al., 2017": 'W. Samek, A. Binder, G. Montavon, S. Lapuschkin, and K.-R. Müller, "Evaluating the visualization of what a deep neural network has learned," IEEE Transactions on Neural Networks and Learning Systems, vol. 28, no. 11, pp. 2660–2673, 2017.',
 "DeYoung et al., 2020": 'J. DeYoung, S. Jain, N. F. Rajani, E. Lehman, C. Xiong, R. Socher, and B. C. Wallace, "ERASER: A benchmark to evaluate rationalized NLP models," in Proc. ACL, 2020, pp. 4443–4458.',
 "Holm, 1979": 'S. Holm, "A simple sequentially rejective multiple test procedure," Scandinavian Journal of Statistics, vol. 6, no. 2, pp. 65–70, 1979.',
}
MAP = {"Abnar and Zuidema, 2020": 51, "Akiba et al., 2019": 52, "Chawla et al., 2002": 29, "Devlin et al., 2019": 54,
       "Ferrag et al., 2020": 4, "Gorishniy et al., 2021": 10, "Hendrycks and Gimpel, 2016": 55, "Jain and Wallace, 2019": 47,
       "Lin et al., 2017": 34, "Loshchilov and Hutter, 2017": 50, "Loshchilov and Hutter, 2019": 53, "Lundberg and Lee, 2017": 14,
       "Ribeiro et al., 2016": 15, "Sharafaldin et al., 2018": 16, "Vaswani et al., 2017": 9, "Wiegreffe and Pinter, 2019": 48,
       "Yang et al., 2016": 24}
txt = re.sub(r'\b([A-Z][A-Za-z\-]+(?: et al\.| and [A-Z][A-Za-z\-]+)?) \[((?:19|20)\d{2})\]', lambda m: f"[{m.group(1)}, {m.group(2)}]", src)
order, numbers = [], {}
def num(k):
    k = k.strip()
    if k not in numbers:
        if k not in MAP and k not in NEW:
            sys.exit(f"UNMAPPED: {k!r}")
        order.append(k); numbers[k] = len(order)
    return numbers[k]
txt = re.sub(r'\[((?:[A-Z][^\[\]]*?(?:19|20)\d{2}[a-z]?)(?:;\s*[^\[\]]*?(?:19|20)\d{2}[a-z]?)*)\]',
             lambda m: "[" + ",".join(str(n) for n in sorted(num(k) for k in m.group(1).split(';'))) + "]", txt)
refs = ["## References", ""]
for i, k in enumerate(order, 1):
    refs += [f"[{i}] {NEW[k] if k in NEW else pool[MAP[k]]}", ""]
txt = txt[:txt.find("## References")] + "\n".join(refs)
open("manuscript/scirep_numbered.md", "w").write(txt)
left = re.findall(r'\[[^\]\d][^\]]*(?:19|20)\d{2}[^\]]*\]', txt)[:3]
print(len(order), "references; leftovers:", left)
