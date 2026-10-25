local t = {}
for i = 1, 300000 do t[i] = i * 2 end
local sum = 0
for round = 1, 4 do
    for i = 1, #t do sum = sum + t[i] end
end
print(sum, #t)
