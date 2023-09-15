import os
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import *
from pathlib import Path
from time import sleep
from numba import cuda
from pygame import mixer

parent_dir = 'D:/im_getting_a_raise/'
mixer.init()
mixer.music.load('D:/im_getting_a_raise/xuehuapiaopiao.mp3')
mixer.music.play(-1)

# Read the text file and get the list of models(model path)0
def get_model():
    with open('D:/im_getting_a_raise/models.txt', 'r') as f:
        modl = f.readlines()
        modl = [i.strip() for i in modl]
        modl = [i.split(', ') for i in modl]
        modl = {i[0]: i[1] for i in modl}
        # list of model paths
        modl_name = list(modl.keys())
        modl_dir = list(modl.values())

    return modl_name, modl_dir

modl_name, modl_dir = get_model()
model_name = modl_name[-1]
model_dir = modl_dir[-1]
  
tokenizer = MarianTokenizer.from_pretrained(model_dir)
model = MarianMTModel.from_pretrained(model_dir)

def translate(text):
    global model_name, model_dir, tokenizer, model
    modl_dir = get_model()[1]
    # skip the model assignment if the model path is the same as the previous one
    if modl_dir[-1] != model_dir:
        model_name = modl_name[-1]
        model_dir = modl_dir[-1]
        tokenizer = MarianTokenizer.from_pretrained(model_dir)
        model = MarianMTModel.from_pretrained(model_dir)
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    out = tgt_text[0]
    output_box.configure(state='normal')
    output_box.delete('1.0', 'end')
    output_box.insert('1.0', out)
    output_box.configure(state='disabled')

def option_1():
    # make clicking on the main window impossible
    root.attributes('-disabled', True)
    window = tk.Toplevel(root)
    window.title('Options')
    window.geometry('400x100')
    window.configure(bg='black')
    # label 'Current translator:'
    current_translator_label = tk.Label(window, text='Current translator:', bg='black', fg='white')
    current_translator_label.grid(row=0, column=0, padx=10, pady=10)
    # combobox
    current_translator = tk.StringVar()
    # set the current translator to the last one in the list
    modl_name, modl_dir = get_model()
    current_translator.set(modl_name[-1])
    translator_combobox = ttk.Combobox(window, textvariable=current_translator, values=modl_name, state='readonly')
    translator_combobox.grid(row=0, column=1, padx=10, pady=10)
    
    # add button 'Add'
    def add():
        # open the text file and get the list of model paths
        modl_name, modl_dir = get_model()
        # opne a file dialog and choose model folder
        file = filedialog.askdirectory(initialdir=parent_dir, title='Select model folder')
        # get the name of the model
        name = file.split('/')[-1].split('.')[0]
        # add the name to the combobox
        translator_combobox['values'] = modl_name + [name]
        # set the current translator to the new one
        current_translator.set(name)
        # save the name and its path to the text file (append to new line without using '\n')
        with open('D:/im_getting_a_raise/models.txt', 'a') as f:
            f.write(name + ', ' + file + '\n')

        # update the list of model paths
        modl_name, modl_dir = get_model()
        print(modl_dir)
        # configure the combobox
        translator_combobox.configure(values=modl_name + [name])

    add_button = tk.Button(window, text='Add', bg='black', fg='white', command=add)
    add_button.grid(row=0, column=2, padx=10, pady=10)

    # button 'Change'
    def change(): 
        modl_name, modl_dir = get_model()
        fkin_name_index = modl_name.index(current_translator.get())
        # remove the model name and its path from the text file
        with open('D:/im_getting_a_raise/models.txt', 'r') as f:
            lines = f.readlines()
        with open('D:/im_getting_a_raise/models.txt', 'w') as f:
            for line in lines:
                if not line.startswith(modl_name[fkin_name_index]):
                    f.write(line)
        # add the model name and its path to the text file
        with open('D:/im_getting_a_raise/models.txt', 'a') as f:
            f.write(modl_name[fkin_name_index] + ', ' + modl_dir[fkin_name_index] + '\n')
        # update the list of model paths
        modl_name, modl_dir = get_model()
        print(modl_name[-1])
        model_name = modl_dir[-1]
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        # configure the combobox
        translator_combobox.configure(values=modl_name)
        window.destroy()
        return model, tokenizer

    change_button = tk.Button(window, text='Change', bg='black', fg='white', command=change)
    change_button.grid(row=1, columnspan=2, padx=10, pady=10)

    root.attributes('-disabled', False)
    window.mainloop()

def option_2():
    pass

# Choose CSV file button
def choose_csv():
    csv_file = filedialog.askopenfilename(initialdir=parent_dir, title='Select CSV file', filetypes=(('CSV files', '*.csv'),))
    csv_entry.delete(0, 'end')
    csv_entry.insert(0, csv_file)

# Translate CSV file
def translate_csv():
    translate_button_csv.configure(state='disabled')
    # get the path of the CSV file
    csv_file = csv_entry.get() 
    # translate the column 'Chingchong' and save it to the column 'Vietnamese'
    def translate_and_save(csv):
        try:
            # Get name of csv file
            csv_name = csv.split('/')[-1]
            # Read the CSV file
            df = pd.read_csv(csv, encoding='utf-8-sig', names=['Chingchong'], header=None)
            df['Vietnamese'] = ''

            for i, row in df.iterrows():
                # Translate the text
                translated = model.generate(**tokenizer(row['Chingchong'], return_tensors="pt", padding=True), max_length=256)
                tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
                out = tgt_text[0]
                # Save the translated text to the column 'Vietnamese'
                df.at[i, 'Vietnamese'] = out
                # Update the progress bar
                progress['value'] = (i + 1) / len(df) * 100
                progress.update()
                progress_label.configure(text=f'Progressing: {i + 1}/{len(df)} lines')
        
        except Exception as e:
            messagebox.showerror('Error', f'Something went wrong: {str(e)}')
            translate_button_csv.configure(state='normal')
        
        # save the translated column to the CSV file
        df.to_csv('translated_' + csv_name, encoding='utf-8-sig', index=False)
        # show a message box when the translation is done
        messagebox.showinfo('Done', 'Translation is complete.')

    # create a progress bar that shows the progress of the translation process based on the number of lines in the CSV file
    progress = ttk.Progressbar(tab2, orient=HORIZONTAL, length=300, mode='determinate')
    progress.grid(row=2, columnspan=3, padx=10, pady=20)
    # and the label 'Progressing: i/n lines'
    progress_label = tk.Label(tab2, text='Progressing...', bg='black', fg='white')
    progress_label.grid(row=3, columnspan=3, padx=10, pady=0)
    # start the translation process
    translate_and_save(csv_file)
    # enable the translate button
    translate_button_csv.configure(state='normal')

# root window
root = tk.Tk()
root.title('Chingchong to Vietnamese')
root.geometry('500x220')
root.configure(bg='black')

# position the window in the center of the screen
windowWidth = root.winfo_reqwidth()
windowHeight = root.winfo_reqheight()
positionRight = int(root.winfo_screenwidth()/2 - windowWidth/2)
positionDown = int(root.winfo_screenheight()/2 - windowHeight/2)
root.geometry("+{}+{}".format(positionRight, positionDown))

# Menu bar
menu = Menu(root)
root.config(menu=menu)

# file
filemenu = Menu(menu, tearoff=0)
filemenu.add_command(label='Exit', command=root.quit)
menu.add_cascade(label='File', menu=filemenu)

# options
optionsmenu = Menu(menu, tearoff=0)
optionsmenu.add_command(label='Change translators...', command=option_1)
optionsmenu.add_command(label='Placeholder', command=option_2)
menu.add_cascade(label='Options', menu=optionsmenu)

# first tab is regular translation and second tab is CSV translation
tab_control = ttk.Notebook(root)
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab_control.add(tab1, text='Regular translation')
tab_control.add(tab2, text='CSV translation')
tab_control.pack(expand=1, fill='both')
sth = ttk.Style()
sth.configure('TNotebook', background='black', foreground='white')
# style for tab
sth.configure('TFrame', background='black')

# tab1

# input label
input_label = tk.Label(tab1, text='Input:', bg='black', fg='white')
input_label.grid(row=0, column=0, padx=30, pady=20)

# a frame for the input box
input_frame = tk.Frame(tab1, bg='black')
input_frame.grid(row=0, column=1, padx=10, pady=20, sticky='ew')

# input box
input_box = tk.Text(input_frame, height=2, width=50, bg='white', font=('Segoe UI', 10, 'normal'))
input_box.pack()

# translate button (in the center)
translate_button = tk.Button(tab1, text='Translate', bg='black', fg='white', command=lambda: translate(input_box.get('1.0', 'end')), width=20)
translate_button.grid(columnspan=2, padx=5, pady=5)

# output label
output_label = tk.Label(tab1, text='Output:', bg='black', fg='white')
output_label.grid(row=2, column=0, padx=30, pady=20)

# a frame for the output box
output_frame = tk.Frame(tab1, bg='black')
output_frame.grid(row=2, column=1, padx=10, pady=20, sticky='ew')

# output box
output_box = tk.Text(output_frame, height=2, width=50, bg='white', font=('Segoe UI', 10, 'normal'), state='disabled')
output_box.pack()

# tab2

# Choose CSV file label
csv_label = tk.Label(tab2, text='Choose CSV file:', bg='black', fg='white')
csv_label.grid(row=0, column=0, padx=30, pady=20)

# a entry for the CSV file path
csv_entry = tk.Entry(tab2, width=30, bg='white', font=('Segoe UI', 10, 'normal'))
csv_entry.grid(row=0, column=1, padx=10, pady=20, sticky='ew')

# choose CSV file button
csv_button = tk.Button(tab2, text='Choose', bg='black', fg='white', command=choose_csv)
csv_button.grid(row=0, column=2, padx=10, pady=20)

# translate button
translate_button_csv = tk.Button(tab2, text='Translate', bg='black', fg='white', command=translate_csv, width=20)
translate_button_csv.grid(columnspan=3, padx=5, pady=5)

root.mainloop()
