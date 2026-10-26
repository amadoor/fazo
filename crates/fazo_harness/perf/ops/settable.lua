local t = {1, 2, 3}; local k = 2
for i = 1, 200000 do
    t[k] = i
    t[k] = i
    t[k] = i
    t[k] = i
    t[k] = i
    t[k] = i
    t[k] = i
    t[k] = i
    t[k] = i
    t[k] = i
end
print(t[k])
