"""draw_svm_2d.py - Minh hoa 2D sieu phang SVM tren data IBM that.
Truc X = MonthlyIncome, truc Y = YearsAtCompany (chuan hoa).
Ve: diem Stay/Leave + duong bien + 2 duong margin + khoanh support vectors.
Output: reports/ibm/svm_hyperplane.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df["Attrition"] = (df["Attrition"] == "Yes").astype(int)
X = df[["MonthlyIncome", "YearsAtCompany"]].values.astype(float)
y = df["Attrition"].values
Xs = StandardScaler().fit_transform(X)

svm = LinearSVC(C=1.0, random_state=42, max_iter=8000)
svm.fit(Xs, y)
print(f"[svm2d] train acc={svm.score(Xs, y):.4f}")

fig, ax = plt.subplots(figsize=(8, 6.5))
fig.patch.set_facecolor("white")

stay = y == 0
leave = y == 1
ax.scatter(Xs[stay, 0], Xs[stay, 1], s=22, c="#1f77b4", alpha=0.55, edgecolors="white",
           linewidths=0.4, label=f"Stay (0), n={(stay).sum()}", zorder=2)
ax.scatter(Xs[leave, 0], Xs[leave, 1], s=34, c="#d62728", alpha=0.8, edgecolors="black",
           linewidths=0.5, label=f"Leave (1), n={(leave).sum()}", zorder=3)

# boundary + margins: w.x + b = {-1, 0, +1}
w = svm.coef_[0]
b = svm.intercept_[0]
xx = np.linspace(-1.6, 3.0, 200)
for level, style, lab, lw in [(0, "-", "Decision boundary (hyperplane)", 2.4),
                              (1, "--", "Margin", 1.2),
                              (-1, "--", None, 1.2)]:
    yy = -(w[0] * xx + b - level) / w[1]
    ax.plot(xx, yy, "k", linestyle=style, lw=lw, label=lab, zorder=5)

# support vectors: gan bien nhat, loc thua de rai deu doc theo bien
d = np.abs(svm.decision_function(Xs))
order = np.argsort(d)[:60]
picked, sup = [], []
for i in order:
    if all(np.linalg.norm(Xs[i] - Xs[j]) > 0.35 for j in picked):
        picked.append(i)
        if len(picked) == 6:
            break
sup = np.array(picked)
ax.scatter(Xs[sup, 0], Xs[sup, 1], s=200, facecolors="none", edgecolors="#f39c12",
           linewidths=2.2, label="Support vectors", zorder=6)
margin_w = 2.0 / np.linalg.norm(w)
ax.text(0.02, 0.02, f"margin width = 2/||w|| = {margin_w:.3f}  (maximized by SVM)",
        transform=ax.transAxes, fontsize=8, color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef9e7", edgecolor="#f39c12"))

ax.set_xlabel("MonthlyIncome (standardized)", fontsize=9)
ax.set_ylabel("YearsAtCompany (standardized)", fontsize=9)
ax.set_title("Linear SVM: hyperplane separates Stay / Leave\n"
             "(margin maximized, boundary defined by support vectors)",
             fontsize=11, fontweight="bold", pad=10)
ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(Xs[:, 1].min() - 0.5, Xs[:, 1].max() + 0.5)
plt.tight_layout()
plt.savefig("reports/ibm/svm_hyperplane.png", dpi=160, facecolor="white")
plt.close()
print("Saved reports/ibm/svm_hyperplane.png")
