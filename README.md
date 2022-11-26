# Proyecto_AML

The required files to run the main.py file are located in illano/Proyecto_AML/Entrega_codigo. These files (val.json, train.json, state_dict.pt, trainer1.py) and the main.py file need to be located on the same level. 

To replicate the evaluation results run the following command:

CUDA_VISIBLE_DEVICES=0 python main.py —mode “test” —path ./ 

To test one example from the validation split run the following line:

CUDA_VISIBLE_DEVICES=0 python main.py —mode “demo” —path ./ 

This command will ask you to input a number between 0 and 10041 which corresponds to the chosen validation example.
