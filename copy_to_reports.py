import shutil, os
src1 = r"C:/Users/Public/graph_architecture.png"
src2 = r"C:/Users/Public/model_comparison_table.png"
dst1 = r"C:/Dự án công nghệ thông tin/hr_causal_ai/reports/ibm/graph_architecture.png"
dst2 = r"C:/Dự án công nghệ thông tin/hr_causal_ai/reports/ibm/model_comparison_table.png"
shutil.copy2(src1, dst1)
shutil.copy2(src2, dst2)
print("Done")
