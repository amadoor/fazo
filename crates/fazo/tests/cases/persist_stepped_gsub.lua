local text = "alpha beta gamma delta"
local out, n = text:gsub("%a+", function(word)
  return word:upper():sub(1, 3) .. #word
end)
print(out, n)
print(("x=1,y=2"):gsub("(%w)=(%d)", "%2%1"))
