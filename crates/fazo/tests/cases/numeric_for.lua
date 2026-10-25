for i = 3, 1 do print("never", i) end
for i = math.maxinteger - 2, math.maxinteger do print(i) end
for i = math.mininteger, math.mininteger + 2 do print(i) end
for i = 0x7ffffffffffffffe, math.maxinteger, 2 do print(i) end
for i = 1, 2, 0.5 do print(i, math.type(i)) end
for i = 1.0, 3 do print(i) end
for i = 3, 1, -1 do print(i) end
for i = 1, 3.7 do print(i, math.type(i)) end
for i = 1, math.huge do print(i) break end
for i = 1, 3 do local j = i local k = 10 print(k, j, i) end
for i = 1, 0 do print("never") end
for i = 0.1, 0.35, 0.1 do print(i) end
