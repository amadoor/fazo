local t = {1, 2, 3}
for i = 1, 200000 do
    t[2] = i
    t[2] = i
    t[2] = i
    t[2] = i
    t[2] = i
    t[2] = i
    t[2] = i
    t[2] = i
    t[2] = i
    t[2] = i
end
print(t[2])
