print(pcall(io.lines, "no_such_file_q.txt"))
print(pcall(io.input, "no_such_file_q.txt"))
print(pcall(io.output, "no_such_dir_q/out.txt"))
