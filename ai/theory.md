If I were to summarize GOPS strategy (as a beginner who keeps losing to AI, and my friends), to win the game, you must:
- Win prizes by the minimum threshold
- Lose prizes by the maximum threshold
- Win higher prizes

Very simple!
However, each decision has levels to it.

Let's imagine we're at the beginning of a 9 card GOPS game. We see a 9 as the prize card. We could take the straight forward method - play a 9 to contest the prize. However, we could also fold - play a 1 to give away the 9. Even though you've given away the highest possible prize, you now have so much more power - your opponent now has a chance to waste their next highest cards against your 9. In this situation, we actually have a 65% chance of winning the game! But what if our opponent predicts this, and plays a 2? This is a drastic loss - we are down 9 points out of a potential 45, and our cards are practically equivalent? Unsurpsingly, our winning chances are now a measly 6%. I'd like to call this bluffing, as you have a similarly worthless 'hand', but still won the pot.

To steal poker terms, we can define it as:
Folding: Playing far underneath the card value
Bluffing: Playing underneath the card value, to catch folders.
Calling: Playing near the card value
Raising: Playing far above the card value

(Quite unoriginal, I know. I also debated on whether to replace folding and bluffing with bluffing and bluff-catching, respectively)

To play effectively, one needs to MIX their actions. Why? Let's say you had a complete table of what you should do at every state. If I knew your table, I could play one card above your move everytime, except for the highest card which I will play my 1. For an N card game, I am guaranteed to win by 1+...+(N-1)-N = 1+...+(N-2)-1, which is positive if N>2. You might say "Obviously you wouldn't know my table, and even if we played multiple times, getting to the exact same state is rare so you'd never get to exploit it", which is valid for small games, but humans won't be able to create a full table, and will instead further simplify their pure strategy, making it even more exploitable.

However, you must also mix actions with the right ratio.

