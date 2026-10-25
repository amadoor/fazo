local sum = 0
for i = 1, 3000000 do
    sum = (sum + i * 3) % 1000000007
end
print(sum)
