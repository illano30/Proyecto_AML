import json
import argparse
import trainer1 as tr
from transformers import RobertaTokenizer
import torch
import os
from os.path import join
from tqdm import tqdm


model = tr.Bert_choice()
model.load_state_dict(torch.load('state_dict.pt'))

model = model.cuda()

tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

def run(args):

    #cargo los datos
    with open(join(args.path, 'val.json')) as f:
        js_data = json.loads(f.read())

    context, answer, choice1, choice2, choice3, choice4 = js_data['ctx'], js_data['label'], js_data['choice1'], js_data['choice2'], js_data['choice3'], js_data['choice4']
    choicess = [choice1, choice2, choice3, choice4]

    #tokenize
    _inputs = [tokenizer([context[i], context[i], context[i], context[i]],
                [choicess[0][i], choicess[1][i], choicess[2][i], choicess[3][i]], 
                return_tensors='pt', padding=True)['input_ids'] for i in range(len(context))]

    _inputs = tr.pad_batch_tensorize_3d(_inputs, pad=0, cuda=False)
    _inputs = _inputs.to(args.device)
    answer = torch.tensor(answer).to(args.device)

    if args.mode == 'test':

        model.eval()

        #pasar por el modelo
        total_ccr = 0
        div = len(_inputs)//3

        for i in tqdm(range(div)):
            labels = answer[i*3 : (i*3) + 3]
            loss, _ids = model(_inputs[i*3 : (i*3) + 3], labels)

            #sacar metrica
            ccr = sum([1 if _ids[i] == labels[i] else 0 for i in range(len(_ids))]) / len(_ids) 
            total_ccr += ccr

        
        return (total_ccr/div)*100

    elif args.mode =='demo':
        
        model.eval()

        ej = input('Ingrese el numero del ejemplo que desea probar (0 a 10041): ')
        ej = int(ej)
        contexto = context[ej]
        choices = [choice1[ej], choice2[ej], choice3[ej], choice4[ej]]
        respuesta = answer[ej]
        loss, prediccion = model(_inputs[ej:ej+1], answer[ej:ej+1])

        return contexto, choices, respuesta, prediccion


parser = argparse.ArgumentParser()
parser.add_argument('--mode', required=True, default='test', help='test/demo')
parser.add_argument('--path', required=True, default='.', help='path of data')

args = parser.parse_args()
args.cuda = torch.cuda.is_available() 
args.device = 'cuda'

rta = run(args)

if args.mode == 'test':
    print('Accuracy = ' + str(rta) + '%')
elif args.mode == 'demo':
    contexto, choices, respuesta, prediccion = rta

    
    print('Usted escogio el siguiente ejemplo: \n', 'Contexto: ', contexto, '\n Opciones: ', choices)
    print('La predicción del modelo fue: ', prediccion, '\n')
    print('Esto corresponde a la siguiente frase: \n')
    print(contexto + ' ' + choices[prediccion])
   