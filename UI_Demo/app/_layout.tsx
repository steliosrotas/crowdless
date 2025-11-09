import { Stack } from "expo-router";

export default function RootLayout() {
  // Disable the native header/title produced by the Stack so pages don't
  // automatically show route names like "index" at the top of the screen.
  return <Stack screenOptions={{ headerShown: false }} />;
}
