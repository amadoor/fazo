local x, acc = 0.5, 0.0
for i = 1, 2000000 do
    x = x * 1.0000001 + 0.25
    acc = acc + x / (i + 1.5)
end
print(string.format("%.10f", acc))
