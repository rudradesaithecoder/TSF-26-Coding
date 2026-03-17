import tkinter as tk

#master
root = tk.Tk()
# setting the windows size
root.geometry("800x800")

frame = tk.Frame(root)
frame.place(relx=0.5, rely=0.5, anchor="center")

# creates main 1st page title
label1 = tk.Label(root, text="Eye-Detect - An Online Parkinson's Disease Screening Tool", font=('Helvetica', 17)) 
label1.place(x=100, y=100)

#creates start test button, and disables it until information is retreived
button = tk.Button(root, text="Start Test", bg='beige', height=1, width=20, bd=5, font=('Helvetica', 10), state="disabled")
button.place(x=310, y=525)

# declaring string variable for storing name and password
name_var=tk.StringVar()
age_var=tk.StringVar()
sex_var=tk.StringVar()

# creates submit functions, and enables start test button
def submit():
    name = name_var.get()
    age = name_var.get()
    sex = name_var.get()

    if name != "" and age != "" and sex != "":
        print(name, age, sex)
        button.config(state="active")
    
# creating a label for name using widget Label
name_label = tk.Label(frame, text = 'First name', font=('calibre',10, 'bold'))
 
# creating a entry for input name using widget Entry
name_entry = tk.Entry(frame,textvariable = name_var, font=('calibre',10,'normal'))
 
# creating a label for age
age_label = tk.Label(frame, text = 'Age', font = ('calibre',10,'bold'))
 
# creating a entry for age
age_entry=tk.Entry(frame, textvariable = age_var, font = ('calibre',10,'normal'))

# creating a label for sex using widget Label
sex_label = tk.Label(frame, text = 'Sex (M/F)', font=('calibre',10, 'bold'))
 
# creating a entry for sex using widget Entry
sex_entry = tk.Entry(frame,textvariable = sex_var, font=('calibre',10,'normal'))

# creating a button using the widget Button that will call the submit function 
sub_btn=tk.Button(frame,text = 'Submit', command = submit)
 
# placing the label and entry in the required position using grid method
name_label.grid(row=0, column=0, padx=10, pady=10)
name_entry.grid(row=0, column=1)

age_label.grid(row=1, column=0, padx=10, pady=10)
age_entry.grid(row=1, column=1)

sex_label.grid(row=2, column=0, padx=10, pady=10)
sex_entry.grid(row=2, column=1)

sub_btn.grid(row=3, column=0, columnspan=2, pady=15)
    
root.mainloop()