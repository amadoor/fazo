local t = {1, 2, 3}
print(#t, #{}, #{n = 1})
t[3] = nil
print(#t)
print(#{1, 2, nil}, #{1, 2, 3, nil})
