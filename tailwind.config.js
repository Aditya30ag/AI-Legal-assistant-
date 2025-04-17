/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{html,js}",'./pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
      },textShadow: {
        'default': '2px 2px 4px rgba(0, 0, 0, 0.3)',
      },
      keyframes: {
        gooey: {
            from: {
                filter: 'blur(20px)',
                transform: 'translate(10%, -10%) skew(0)',
            },
            to: {
                filter: 'blur(30px)',
                transform: 'translate(-10%, 10%) skew(-12deg)',
              },
        }
      },
    },
  },
  plugins: [],
}