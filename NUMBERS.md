## Numbers

Here are some numbers not in the paper that might be useful for benchmarks

Cos Sim number of Figure 2: Prompt extraction quality vs. the number of LLM outputs provided to the inverter.
```
       llama    gpt3_5   mistral     gemma
1   0.929942  0.887204  0.897990  0.862491
2   0.940565  0.895099  0.907304  0.876809
4   0.950718  0.906559  0.915764  0.892883
8   0.956547  0.914244  0.919632  0.902821
16  0.960991  0.921434  0.924757  0.910870
32  0.965134  0.927007  0.929447  0.918164
64  0.966564  0.929895  0.933782  0.923652
```

BLEU number of Figure 2: Prompt extraction quality vs. the number of LLM outputs provided to the inverter.
```
       llama    gpt3_5   mistral     gemma
1   0.299479  0.156946  0.135473  0.071363
2   0.365731  0.190263  0.171250  0.097952
4   0.422332  0.230717  0.210595  0.141855
8   0.470836  0.263421  0.240160  0.186674
16  0.516508  0.287414  0.267906  0.222852
32  0.552837  0.314654  0.295738  0.271834
64  0.569457  0.334392  0.324195  0.309886
```