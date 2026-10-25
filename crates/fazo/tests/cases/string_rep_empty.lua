print(pcall(string.rep, "", math.maxinteger))
print(pcall(string.rep, "", 1000000000, ""))
print(string.rep("", 3, "x") == "", string.rep("ab", 3, ","), string.rep("x", 0, "y"), string.rep("x", -1))
print(#string.rep("", 0), #string.rep("", 5, ""), string.rep("", 2, "-"))
