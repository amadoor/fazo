local f = ipairs{}
local k, v = f({[math.mininteger] = 10}, math.maxinteger)
print(k, v)
k, v = f({[math.mininteger] = 10}, k)
print(k)
for i, x in ipairs({5, 6}) do print(i, x) end
