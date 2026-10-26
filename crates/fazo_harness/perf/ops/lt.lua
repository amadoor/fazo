local x, y = 1, 2
for i = 1, 200000 do
    if y < x then x = y end
    if y < x then x = y end
    if y < x then x = y end
    if y < x then x = y end
    if y < x then x = y end
    if y < x then x = y end
    if y < x then x = y end
    if y < x then x = y end
    if y < x then x = y end
    if y < x then x = y end
end
print(x)
