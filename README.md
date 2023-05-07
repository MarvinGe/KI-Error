# KI-Error

Hi, I am a beginner in using tensoflow and i built a classification model relying heavily on tf_agent.
Some time ago i increased the number of used datasets and since then i get the following error message.

Error Message:
"W tensorflow/core/framework/op_kernel.cc:1830] OP_REQUIRES failed at gather_nd_op.cc:48 : INVALID_ARGUMENT: indices[4] = [4, 200, 66] does not index into param shape [10,200,200], node name: GatherNd"

This error always occurs after step 8 when running the code.
Changing the number of training loops and number of datasets changes also the step count till the error occurs. Coudln't to find a direct correlation between those two yet as well as the cause why it occurs in the first place.

Error reproducable: yes
Used python: 3.11.2
Used code: https://github.com/MarvinGe/KI-Error/blob/main/Main.py
Used packages: https://github.com/MarvinGe/KI-Error/blob/main/requirements.txt
Detailled error description: https://github.com/MarvinGe/KI-Error/blob/main/Error_Description.png

Tags:
help_request, tf_agents, graph
