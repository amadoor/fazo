print(io.open("no_such_dir_q/a.txt"))
print(select(3, io.open("no_such_dir_q/a.txt", "w")))
print(os.remove("no_such_file_q.txt"))
print(os.rename("no_such_file_q.txt", "other_q.txt"))
