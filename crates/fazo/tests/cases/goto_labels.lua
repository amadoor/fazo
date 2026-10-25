for i = 1, 3 do
  if i == 2 then goto continue end
  print("i", i)
  ::continue::
end
do
  local n = 0
  ::top::
  n = n + 1
  if n < 3 then goto top end
  print("n", n)
end
for i = 1, 2 do
  for j = 1, 2 do
    if j == 2 then goto next_i end
    print(i, j)
  end
  ::next_i::
end
do
  goto skip
  print("skipped")
  ::skip::
  print("after skip")
end
